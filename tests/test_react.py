from __future__ import annotations

from agent_core.config import AgentConfig
from agent_core.llm.base import LLMResponse
from agent_core.llm.fake import FakeBackend
from agent_core.messages import ToolCall
from agent_core.strategies.react import ReActStrategy
from agent_core.tools import ToolRegistry


async def test_reacts_to_a_tool_result_before_answering(tools: ToolRegistry) -> None:
    backend = FakeBackend(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})]),
            LLMResponse(content="the tool said: echo: hi"),
        ]
    )
    trace = await ReActStrategy().run("t1", "say hi", "system", backend, tools, AgentConfig())

    assert trace.final_answer == "the tool said: echo: hi"
    assert trace.llm_calls == 2
    assert trace.tool_calls == 1
    assert not trace.hit_step_limit


async def test_answers_immediately_when_no_tool_needed(tools: ToolRegistry) -> None:
    backend = FakeBackend([LLMResponse(content="42")])
    trace = await ReActStrategy().run("t1", "what is 42", "system", backend, tools, AgentConfig())

    assert trace.final_answer == "42"
    assert trace.llm_calls == 1
    assert trace.tool_calls == 0


async def test_hits_step_limit_when_model_keeps_calling_tools(tools: ToolRegistry) -> None:
    def always_call_tool(messages: object, tool_schemas: object) -> LLMResponse:
        return LLMResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "again"})])

    backend = FakeBackend(always_call_tool)
    config = AgentConfig(max_steps=3)
    trace = await ReActStrategy().run("t1", "loop forever", "system", backend, tools, config)

    assert trace.hit_step_limit is True
    assert trace.final_answer == ""
    assert trace.llm_calls == 3


async def test_failed_tool_call_is_visible_to_the_model(tools: ToolRegistry) -> None:
    seen_error_observation = False

    def script(messages: list[object], tool_schemas: object) -> LLMResponse:
        nonlocal seen_error_observation
        from agent_core.messages import Message

        for m in messages:
            assert isinstance(m, Message)
            if m.role == "tool" and "boom" in m.content:
                seen_error_observation = True
        if seen_error_observation:
            return LLMResponse(content="the tool failed")
        return LLMResponse(tool_calls=[ToolCall(id="1", name="fail", arguments={})])

    backend = FakeBackend(script)
    trace = await ReActStrategy().run(
        "t1", "call the failing tool", "system", backend, tools, AgentConfig()
    )

    assert trace.final_answer == "the tool failed"
    assert trace.failed_tool_calls == 1
