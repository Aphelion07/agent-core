"""The backend contract.

A strategy calls ``chat`` and gets back either free text (the model is done)
or tool calls (the model wants an observation before it continues). Nothing
above this layer knows whether it is talking to a local Ollama daemon, an
OpenAI-compatible endpoint, or the scripted fake used by the test suite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ..messages import Message, ToolCall


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMBackend(ABC):
    name: str

    @abstractmethod
    async def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        """One completion turn. ``tools`` may be empty to force plain text."""

    async def aclose(self) -> None:  # noqa: B027 - optional, not abstract
        """Release connections. Safe to call more than once."""
