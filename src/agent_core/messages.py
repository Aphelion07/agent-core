"""Wire-agnostic conversation state.

Every backend (Ollama, an OpenAI-compatible endpoint, the in-process fake used
by tests) translates to and from this shape at its own boundary. Nothing above
that boundary - strategies, the context trimmer, the trace - knows which
backend it is talking to.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    # Only set on role="assistant" messages that requested tool calls.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Only set on role="tool" messages, linking the result back to a ToolCall.id.
    tool_call_id: str | None = None
    name: str | None = None

    def approx_tokens(self) -> int:
        """Cheap token estimate; good enough for context-budget trimming."""
        return max(1, len(self.content) // 4)
