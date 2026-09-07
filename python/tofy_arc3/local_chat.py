# SPDX-License-Identifier: MIT-0
"""Local-chat baseline with fixed decoding settings for ARC-AGI-3."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from .frame_memory import (
    PALETTE,
    Frame,
    encode_frame,
    summarize_components,
    summarize_frame_change,
)

_MAX_RESPONSE_BYTES = 1 << 20
_FENCED_JSON = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)
_FRAME_FORMATS = {"raw_hex", "compact"}
_SAMPLING_FIELDS = {"temperature", "top_p", "top_k", "min_p"}
_CONTEXT_SAFETY_TOKENS = 128
_TRANSITION_LEDGER_ROWS = 8

SYSTEM_PROMPT = """You are an ARC-AGI-3 game-playing policy. Infer the visible game's rules and goal only from the observation sequence and the results of prior actions. Aim to make progress and reach WIN while using actions efficiently. The board is provided as lossless text; you have no vision, code-execution, external-data, or game-source tools.

The available decision actions use the normal ARC action protocol: ACTION1=up input, ACTION2=down input, ACTION3=left input, ACTION4=right input, ACTION5=interact, ACTION6=coordinate input, and ACTION7=undo. RESET is handled by the caller and cannot be selected here. These are interface meanings only: an action's actual game effect can vary, and directional inputs do not necessarily move a player. Palette symbols 0-F are categorical values with no assumed color or object meaning.

Choose one listed available action. Reply with exactly one JSON object. For ACTION6 use {"action_id":6,"x":0,"y":0}, with x and y in 0..63. For every other action use {"action_id":N} and omit x and y. Do not include prose or extra keys."""


class LocalChatError(RuntimeError):
    """The local endpoint or its assistant response violated the contract."""


@dataclass(frozen=True, slots=True)
class _TransitionBefore:
    last_frame: Frame
    levels_completed: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class _PendingTransition:
    session: tuple[str, str]
    action: str
    before: _TransitionBefore


def validate_sampling(sampling: object) -> dict[str, int | float]:
    """Return a validated copy of supported llama-server sampling fields."""
    if not isinstance(sampling, dict):
        raise TypeError("sampling must be an object")
    unknown = set(sampling) - _SAMPLING_FIELDS
    if unknown:
        raise ValueError(f"sampling has unknown fields: {sorted(map(str, unknown))}")
    result: dict[str, int | float] = {}
    for name, value in sampling.items():
        if name == "top_k":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("sampling.top_k must be a nonnegative integer")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (name == "temperature" and not 0 <= value <= 2)
            or (name == "top_p" and not 0 < value <= 1)
            or (name == "min_p" and not 0 <= value <= 1)
        ):
            raise ValueError(f"sampling.{name} is outside its allowed finite range")
        result[name] = value
    return result


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _completion_url(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("endpoint must be a non-empty URL")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP(S) loopback URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint credentials are not allowed")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint query strings and fragments are not allowed")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint port is invalid") from exc
    host = parsed.hostname.lower()
    try:
        host = ipaddress.ip_address(host).compressed
    except ValueError as exc:
        raise ValueError("endpoint host must be numeric loopback") from exc
    if host not in {"127.0.0.1", "::1"}:
        raise ValueError("endpoint host must be 127.0.0.1 or ::1")
    path = parsed.path
    if path in {"", "/"}:
        path = "/v1/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _integer(value: Any, field: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be in 0..{maximum}")
    return value


def _observation_text(
    observation: dict[str, Any],
    *,
    require_actions: bool = True,
    frame_format: str = "raw_hex",
) -> tuple[tuple[str, str], str, list[int], str]:
    if not isinstance(observation, dict):
        raise TypeError("observation must be a JSON object")
    game_id = observation.get("game_id")
    guid = observation.get("guid")
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("observation game_id must be a non-empty string")
    if not isinstance(guid, str) or not guid:
        raise ValueError("observation guid must be a non-empty string")
    state = observation.get("state")
    if state not in {"NOT_STARTED", "NOT_FINISHED", "WIN", "GAME_OVER"}:
        raise ValueError("observation state is invalid")
    levels_completed = _integer(observation.get("levels_completed"), "levels_completed")
    win_levels = _integer(observation.get("win_levels"), "win_levels")
    full_reset = observation.get("full_reset")
    if not isinstance(full_reset, bool):
        raise TypeError("observation full_reset must be a boolean")
    available = observation.get("available_actions")
    if not isinstance(available, list) or (require_actions and not available):
        requirement = "a non-empty list" if require_actions else "a list"
        raise ValueError(f"observation available_actions must be {requirement}")
    for index, action_id in enumerate(available):
        _integer(action_id, f"available_actions[{index}]", maximum=7)
    if len(set(available)) != len(available):
        raise ValueError("observation available_actions must be unique")
    if require_actions:
        available = [action_id for action_id in available if action_id != 0]
        if not available:
            raise ValueError("observation has no available non-reset action")

    frames = observation.get("frame")
    if not isinstance(frames, list) or not frames:
        raise ValueError("observation frame must contain at least one layer")
    rendered: list[str] = []
    for layer_index, layer in enumerate(frames):
        if not isinstance(layer, list) or len(layer) != 64:
            raise ValueError(f"frame layer {layer_index} must have 64 rows")
        rows: list[str] = []
        values_by_row: list[list[int]] = []
        for row_index, row in enumerate(layer):
            if not isinstance(row, list) or len(row) != 64:
                raise ValueError(
                    f"frame layer {layer_index} row {row_index} must have 64 cells"
                )
            values = [
                _integer(value, f"frame[{layer_index}][{row_index}]", maximum=15)
                for value in row
            ]
            values_by_row.append(values)
            rows.append("".join(format(value, "X") for value in values))
        if frame_format == "raw_hex":
            rendered.append(f"layer {layer_index}:\n" + "\n".join(rows))
        else:
            components = summarize_components(values_by_row, max_components=32)
            component_lines = [
                (
                    f"components layer {layer_index}: total={components.total_components} "
                    f"shown={len(components.components)} "
                    f"truncated={components.truncated_count}"
                )
            ]
            component_lines.extend(
                (
                    f"color={PALETTE[component.color]} area={component.area} "
                    f"bbox={component.bbox} centroid={component.centroid}"
                )
                for component in components.components
            )
            rendered.append(
                f"layer {layer_index} compact:\n{encode_frame(values_by_row)}\n"
                + "\n".join(component_lines)
            )

    if frame_format == "raw_hex":
        frame_legend = (
            "Each frame row is 64 lossless hexadecimal palette symbols (0-F); x is the "
            "column and y is the row.\n"
        )
    else:
        frame_legend = (
            "Compact frame legend: each header gives x/y origin, size, and categorical "
            "palette; a bare 64-symbol line is one raw row; Pn:<row> repeats a raw "
            "row n times; Rn:<runs> repeats an RLE row n times; comma-separated "
            "symbol*count entries are horizontal runs (a bare symbol has count 1); n "
            "is the number of decoded rows.\n"
        )
    text = (
        "Visible observation:\n"
        f"state={state}\n"
        f"levels_completed={levels_completed}\n"
        f"win_levels={win_levels}\n"
        f"full_reset={json.dumps(full_reset)}\n"
        f"available_actions={json.dumps(available, separators=(',', ':'))}\n"
        + frame_legend
        + "\n".join(rendered)
    )
    return (game_id, guid), text, available, state


def _parse_action(content: str, available: list[int]) -> dict[str, int]:
    candidate = content.strip()
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(candidate, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LocalChatError("assistant response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise LocalChatError("assistant action must be a JSON object")
    action_id = value.get("action_id")
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise LocalChatError("assistant action_id must be an integer")
    if action_id not in available:
        raise LocalChatError(f"assistant chose unavailable action {action_id}")
    expected_keys = {"action_id", "x", "y"} if action_id == 6 else {"action_id"}
    if set(value) != expected_keys:
        raise LocalChatError("assistant action has missing or extra keys")
    result = {"action_id": action_id}
    if action_id == 6:
        try:
            result["x"] = _integer(value["x"], "assistant ACTION6 x", maximum=63)
            result["y"] = _integer(value["y"], "assistant ACTION6 y", maximum=63)
        except ValueError as exc:
            raise LocalChatError(str(exc)) from exc
    return result


def _action_response_format(available: list[int]) -> dict[str, Any]:
    branches: list[dict[str, Any]] = []
    simple = [action_id for action_id in available if action_id != 6]
    if simple:
        branches.append(
            {
                "type": "object",
                "properties": {"action_id": {"type": "integer", "enum": simple}},
                "required": ["action_id"],
                "additionalProperties": False,
            }
        )
    if 6 in available:
        coordinate = {"type": "integer", "minimum": 0, "maximum": 63}
        branches.append(
            {
                "type": "object",
                "properties": {
                    "action_id": {"type": "integer", "const": 6},
                    "x": coordinate,
                    "y": coordinate,
                },
                "required": ["action_id", "x", "y"],
                "additionalProperties": False,
            }
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "arc_action",
            "strict": True,
            "schema": {"anyOf": branches},
        },
    }


class LocalChatAgent:
    """Choose validated actions through a loopback OpenAI-compatible endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        seed: int = 0,
        max_tokens: int = 1024,
        timeout: float = 30,
        history_turns: int = 4,
        frame_format: str = "raw_hex",
        sampling: object | None = None,
        context_size: int | None = None,
        factual_transition_ledger: bool = False,
    ) -> None:
        self.endpoint = _completion_url(endpoint)
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        self.model = model
        self.seed = _integer(seed, "seed")
        self.max_tokens = _integer(max_tokens, "max_tokens")
        if self.max_tokens == 0:
            raise ValueError("max_tokens must be positive")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)
        self.history_turns = _integer(history_turns, "history_turns")
        if not isinstance(frame_format, str) or frame_format not in _FRAME_FORMATS:
            raise ValueError("frame_format must be raw_hex or compact")
        self.frame_format = frame_format
        self.sampling = {} if sampling is None else validate_sampling(sampling)
        self.context_size = (
            None if context_size is None else _integer(context_size, "context_size")
        )
        if self.context_size == 0:
            raise ValueError("context_size must be positive")
        if not isinstance(factual_transition_ledger, bool):
            raise TypeError("factual_transition_ledger must be a boolean")
        self.factual_transition_ledger = factual_transition_ledger
        self._history: list[tuple[str, str | None, str | None]] = []
        self._session: tuple[str, str] | None = None
        self._transitions: deque[str] = deque(maxlen=_TRANSITION_LEDGER_ROWS)
        self._pending_transition: _PendingTransition | None = None
        self._open: Callable[..., Any] = build_opener(
            ProxyHandler({}), _NoRedirect()
        ).open

    def _clear_transition_ledger(self) -> None:
        self._transitions.clear()
        self._pending_transition = None

    def _advance_session(
        self, session: tuple[str, str], observation: dict[str, Any]
    ) -> None:
        if session != self._session:
            self._history.clear()
            self._clear_transition_ledger()
            self._session = session
        elif self.factual_transition_ledger and observation["full_reset"]:
            self._clear_transition_ledger()

    def _fold_pending_transition(
        self, session: tuple[str, str], observation: dict[str, Any]
    ) -> None:
        if not self.factual_transition_ledger or self._pending_transition is None:
            return
        pending = self._pending_transition
        if pending.session != session:
            self._pending_transition = None
            return
        frames = observation["frame"]
        change = summarize_frame_change(pending.before.last_frame, frames[-1])
        outcome = {
            "last_frame_changed_pixel_count": change.changed_pixel_count,
            "last_frame_changed_bbox": change.bbox,
            "state_after": observation["state"],
            "levels_completed_delta": (
                observation["levels_completed"] - pending.before.levels_completed
            ),
            "full_reset_after": observation["full_reset"],
            "frame_count_before": pending.before.frame_count,
            "frame_count_after": len(frames),
        }
        self._transitions.append(
            f'{{"action":{pending.action},"outcome":'
            + json.dumps(outcome, separators=(",", ":"))
            + "}"
        )
        self._pending_transition = None

    def _current_observation(self, observation_text: str) -> str:
        if not self.factual_transition_ledger or not self._transitions:
            return observation_text
        return (
            observation_text
            + "\n\nExperienced factual transition ledger (oldest to newest):\n"
            + "\n".join(self._transitions)
        )

    def _transition_before(
        self, observation: dict[str, Any]
    ) -> _TransitionBefore | None:
        if not self.factual_transition_ledger:
            return None
        frames = observation["frame"]
        return _TransitionBefore(
            last_frame=tuple(tuple(row) for row in frames[-1]),
            levels_completed=observation["levels_completed"],
            frame_count=len(frames),
        )

    def _budget_request(
        self, path: str, payload: dict[str, Any], deadline: float
    ) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("context budgeting exceeded the decision deadline")
        parsed = urlsplit(self.endpoint)
        request = Request(
            urlunsplit((parsed.scheme, parsed.netloc, path, "", "")),
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with self._open(request, timeout=remaining) as response:
                status = getattr(response, "status", None)
                if status is not None and not 200 <= status < 300:
                    raise LocalChatError(f"context endpoint returned HTTP {status}")
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise LocalChatError(f"context endpoint returned HTTP {exc.code}") from exc
        except TimeoutError:
            raise
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise exc.reason from exc
            raise LocalChatError(f"context request failed: {exc}") from exc
        except OSError as exc:
            raise LocalChatError(f"context request failed: {exc}") from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LocalChatError("context response exceeds the size limit")
        try:
            value = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocalChatError("context endpoint returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise LocalChatError("context endpoint returned a non-object response")
        if time.monotonic() > deadline:
            raise TimeoutError("context budgeting exceeded the decision deadline")
        return value

    def _fit_messages(
        self, current: str, deadline: float
    ) -> tuple[list[dict[str, str]], int, int]:
        groups: list[list[dict[str, str]]] = []
        for prior, action, terminal in self._history:
            group = [{"role": "user", "content": prior}]
            if action is not None:
                group.append({"role": "assistant", "content": action})
            if terminal is not None:
                group.append({"role": "user", "content": terminal})
            groups.append(group)
        for evicted in range(len(groups) + 1):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(item for group in groups[evicted:] for item in group)
            messages.append({"role": "user", "content": current})
            applied = self._budget_request(
                "/apply-template", {"messages": messages}, deadline
            )
            prompt = applied.get("prompt")
            if not isinstance(prompt, str):
                raise LocalChatError("apply-template response has no string prompt")
            tokenized = self._budget_request(
                "/tokenize",
                {"content": prompt, "add_special": False, "parse_special": True},
                deadline,
            )
            tokens = tokenized.get("tokens")
            if not isinstance(tokens, list) or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in tokens
            ):
                raise LocalChatError("tokenize response has no valid token ID list")
            estimated = len(tokens)
            if (
                estimated + self.max_tokens + _CONTEXT_SAFETY_TOKENS
                <= self.context_size
            ):
                return messages, evicted, estimated
        raise LocalChatError(
            "system prompt and current lossless observation exceed context_size"
        )

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        decision_started = time.perf_counter()
        session, observation_text, available, state = _observation_text(
            observation, frame_format=self.frame_format
        )
        if state != "NOT_FINISHED":
            raise ValueError("choose_action requires a NOT_FINISHED observation")
        self._advance_session(session, observation)
        if not observation["full_reset"]:
            self._fold_pending_transition(session, observation)
        current_observation = self._current_observation(observation_text)
        transition_before = self._transition_before(observation)

        deadline = (
            None if self.context_size is None else time.monotonic() + self.timeout
        )
        if self.context_size is None:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for prior_observation, prior_action, terminal_feedback in self._history:
                messages.append({"role": "user", "content": prior_observation})
                if prior_action is not None:
                    messages.append({"role": "assistant", "content": prior_action})
                if terminal_feedback is not None:
                    messages.append({"role": "user", "content": terminal_feedback})
            messages.append({"role": "user", "content": current_observation})
            evicted, estimated = 0, None
        else:
            preflight_started = time.perf_counter()
            messages, evicted, estimated = self._fit_messages(
                current_observation, deadline
            )
            preflight_seconds = time.perf_counter() - preflight_started
        history_before = len(self._history)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self.seed,
            "stream": False,
            "response_format": _action_response_format(available),
        }
        payload.update(self.sampling)
        request = Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            remaining = (
                self.timeout if deadline is None else deadline - time.monotonic()
            )
            if deadline is not None and remaining <= 0:
                raise TimeoutError("context budgeting exceeded the decision deadline")
            with self._open(request, timeout=remaining) as response:
                status = getattr(response, "status", None)
                if status is None and hasattr(response, "getcode"):
                    status = response.getcode()
                if status is not None and not 200 <= status < 300:
                    raise LocalChatError(f"chat endpoint returned HTTP {status}")
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise LocalChatError(f"chat endpoint returned HTTP {exc.code}") from exc
        except TimeoutError:
            raise
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise exc.reason from exc
            raise LocalChatError(f"chat request failed: {exc}") from exc
        except OSError as exc:
            raise LocalChatError(f"chat request failed: {exc}") from exc
        latency = time.perf_counter() - started
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LocalChatError("chat response exceeds the size limit")
        try:
            envelope = json.loads(raw.decode("utf-8"))
            choices = envelope["choices"]
            message = choices[0]["message"]
            content = message["content"]
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise LocalChatError(
                "chat endpoint returned an invalid completion response"
            ) from exc
        if not isinstance(content, str):
            raise LocalChatError("assistant response content must be text")
        action = _parse_action(content, available)
        canonical_action = json.dumps(action, separators=(",", ":"))
        usage = envelope.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        actual = usage.get("prompt_tokens")
        if self.context_size is not None:
            if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
                raise LocalChatError("completion did not report valid prompt_tokens")
            if abs(actual - estimated) > _CONTEXT_SAFETY_TOKENS:
                raise LocalChatError("actual prompt usage drifted from token estimate")
            if actual + self.max_tokens + _CONTEXT_SAFETY_TOKENS > self.context_size:
                raise LocalChatError("actual prompt usage exceeded context_size")
            del self._history[:evicted]
        if self.history_turns:
            self._history.append((observation_text, canonical_action, None))
            del self._history[: -self.history_turns]
        if transition_before is not None:
            self._pending_transition = _PendingTransition(
                session=session,
                action=canonical_action,
                before=transition_before,
            )
        telemetry = {
            "elapsed_seconds": latency,
            "token_usage": usage,
            "raw_assistant_response": content,
        }
        if self.context_size is not None:
            telemetry["context_budget"] = {
                "context_size": self.context_size,
                "safety_tokens": _CONTEXT_SAFETY_TOKENS,
                "max_tokens": self.max_tokens,
                "estimated_prompt_tokens": estimated,
                "actual_prompt_tokens": actual,
                "history_turns_kept": history_before - evicted,
                "history_turns_evicted": evicted,
                "preflight_seconds": preflight_seconds,
                "total_decision_seconds": time.perf_counter() - decision_started,
            }
        if "reasoning_content" in message:
            telemetry["reasoning_content"] = message["reasoning_content"]
        return {
            **action,
            "telemetry": telemetry,
        }

    def observe_terminal(self, observation: dict[str, Any]) -> None:
        """Record WIN/GAME_OVER feedback without making an HTTP request."""
        session, observation_text, _, state = _observation_text(
            observation,
            require_actions=False,
            frame_format=self.frame_format,
        )
        if state not in {"WIN", "GAME_OVER"}:
            raise ValueError("observe_terminal requires a terminal observation")
        self._advance_session(session, observation)
        if not observation["full_reset"]:
            self._fold_pending_transition(session, observation)
        if self.history_turns:
            if self._history and self._history[-1][1] is not None:
                prior_observation, prior_action, _ = self._history[-1]
                self._history[-1] = (
                    prior_observation,
                    prior_action,
                    observation_text,
                )
            else:
                self._history.append((observation_text, None, None))
                del self._history[: -self.history_turns]

    def close(self) -> None:
        """No persistent transport is held by this client."""
