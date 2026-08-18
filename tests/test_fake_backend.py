from __future__ import annotations

import pytest

from agent_core.llm.base import LLMResponse
from agent_core.llm.fake import FakeBackend
from agent_core.messages import Message


async def test_plays_back_scripted_responses_in_order() -> None:
    backend = FakeBackend([LLMResponse(content="first"), LLMResponse(content="second")])
    first = await backend.chat([Message(role="user", content="hi")], [])
    second = await backend.chat([Message(role="user", content="hi")], [])
    assert first.content == "first"
    assert second.content == "second"
    assert backend.call_count == 2


async def test_raises_when_script_exhausted() -> None:
    backend = FakeBackend([LLMResponse(content="only")])
    await backend.chat([], [])
    with pytest.raises(IndexError):
        await backend.chat([], [])


async def test_callable_script_sees_full_conversation() -> None:
    def script(messages: list[Message], tools: list[dict[str, object]]) -> LLMResponse:
        return LLMResponse(content=f"saw {len(messages)} messages, {len(tools)} tools")

    backend = FakeBackend(script)
    response = await backend.chat(
        [Message(role="system", content="s"), Message(role="user", content="u")],
        [{"type": "function", "function": {"name": "x"}}],
    )
    assert response.content == "saw 2 messages, 1 tools"
    assert backend.history[-1][0][0].role == "system"
