from __future__ import annotations

from agent_core.config import AgentConfig
from agent_core.llm.base import LLMResponse
from agent_core.llm.fake import FakeBackend
from agent_core.messages import ToolCall
from agent_core.strategies.reflexion import ReflexionStrategy
from agent_core.tools import ToolRegistry


async def test_keeps_candidate_when_verdict_says_correct(tools: ToolRegistry) -> None:
    backend = FakeBackend(
        [
            LLMResponse(content="candidate answer"),
            LLMResponse(
                tool_calls=[ToolCall(id="v", name="submit_verdict", arguments={"correct": True})]
            ),
        ]
    )
    trace = await ReflexionStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=2)
    )

    assert trace.final_answer == "candidate answer"
    assert trace.llm_calls == 2


async def test_replaces_candidate_when_verdict_says_incorrect(tools: ToolRegistry) -> None:
    backend = FakeBackend(
        [
            LLMResponse(content="wrong answer"),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="v",
                        name="submit_verdict",
                        arguments={"correct": False, "corrected_answer": "right answer"},
                    )
                ]
            ),
        ]
    )
    trace = await ReflexionStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=2)
    )

    assert trace.final_answer == "right answer"


async def test_skips_critique_when_budget_is_one(tools: ToolRegistry) -> None:
    backend = FakeBackend([LLMResponse(content="only answer")])
    trace = await ReflexionStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=1)
    )

    assert trace.final_answer == "only answer"
    assert trace.llm_calls == 1


async def test_hits_step_limit_without_attempting_critique(tools: ToolRegistry) -> None:
    def always_call_tool(messages: object, tool_schemas: object) -> LLMResponse:
        return LLMResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})])

    backend = FakeBackend(always_call_tool)
    trace = await ReflexionStrategy().run(
        "t1", "loop forever", "system", backend, tools, AgentConfig(max_steps=2)
    )

    assert trace.hit_step_limit is True
    assert trace.llm_calls == 1


async def test_keeps_candidate_when_model_skips_submit_verdict(tools: ToolRegistry) -> None:
    backend = FakeBackend(
        [LLMResponse(content="candidate"), LLMResponse(content="looks fine to me")]
    )
    trace = await ReflexionStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=2)
    )

    assert trace.final_answer == "candidate"
