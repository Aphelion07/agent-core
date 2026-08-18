"""Deterministic, in-process backend. No network, no GPU.

Two ways to script it: a fixed list of responses played back in order, or a
callable that inspects the conversation so far and decides what to answer.
The callable form is what the test suite uses to assert on exactly which
messages a strategy sent - e.g. that a tool result was appended before the
next call, or that the planning phase used a different tool list than
execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..messages import Message
from .base import LLMBackend, LLMResponse

ScriptFunc = Callable[[list[Message], list[dict[str, Any]]], LLMResponse]


class FakeBackend(LLMBackend):
    name = "fake"

    def __init__(self, script: list[LLMResponse] | ScriptFunc) -> None:
        self._script = script
        self._calls = 0
        self.history: list[tuple[list[Message], list[dict[str, Any]]]] = []

    @property
    def call_count(self) -> int:
        return self._calls

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        self.history.append((list(messages), list(tools)))
        if callable(self._script):
            response = self._script(messages, tools)
        else:
            if self._calls >= len(self._script):
                raise IndexError(
                    f"FakeBackend script exhausted after {self._calls} calls "
                    f"(only {len(self._script)} responses scripted)"
                )
            response = self._script[self._calls]
        self._calls += 1
        return response
