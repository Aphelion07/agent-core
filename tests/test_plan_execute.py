from __future__ import annotations

from agent_core.config import AgentConfig
from agent_core.llm.base import LLMResponse
from agent_core.llm.fake import FakeBackend
from agent_core.messages import ToolCall
from agent_core.strategies.plan_execute import PlanExecuteStrategy
from agent_core.tools import ToolRegistry


async def test_executes_plan_steps_in_order_then_answers(tools: ToolRegistry) -> None:
    backend = FakeBackend(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="p", name="submit_plan", arguments={"steps": ["find x", "compute y"]}
                    )
                ]
            ),
            LLMResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})]),
            LLMResponse(content="y is known"),
            LLMResponse(content="final answer: 42"),
        ]
    )
    trace = await PlanExecuteStrategy().run(
        "t1", "find x then compute y", "system", backend, tools, AgentConfig(max_steps=4)
    )

    assert trace.final_answer == "final answer: 42"
    assert trace.llm_calls == 4
    assert trace.tool_calls == 1
    assert not trace.hit_step_limit


async def test_degrades_to_single_step_plan_without_submit_plan_call(tools: ToolRegistry) -> None:
    backend = FakeBackend(
        [
            LLMResponse(content="thinking out loud instead of calling submit_plan"),
            LLMResponse(content="did the step"),
            LLMResponse(content="final answer"),
        ]
    )
    trace = await PlanExecuteStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=3)
    )

    assert trace.final_answer == "final answer"
    assert trace.llm_calls == 3


async def test_uses_planning_response_as_answer_when_budget_is_one(tools: ToolRegistry) -> None:
    backend = FakeBackend([LLMResponse(content="direct answer")])
    trace = await PlanExecuteStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=1)
    )

    assert trace.final_answer == "direct answer"
    assert not trace.hit_step_limit
    assert trace.llm_calls == 1


async def test_hits_step_limit_when_budget_is_one_and_planning_call_had_no_content(
    tools: ToolRegistry,
) -> None:
    backend = FakeBackend(
        [LLMResponse(tool_calls=[ToolCall(id="p", name="submit_plan", arguments={"steps": ["a"]})])]
    )
    trace = await PlanExecuteStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=1)
    )

    assert trace.hit_step_limit is True
    assert trace.final_answer == ""


async def test_invalid_plan_arguments_degrade_to_single_step(tools: ToolRegistry) -> None:
    backend = FakeBackend(
        [
            LLMResponse(tool_calls=[ToolCall(id="p", name="submit_plan", arguments={"steps": []})]),
            LLMResponse(content="did it anyway"),
            LLMResponse(content="final"),
        ]
    )
    trace = await PlanExecuteStrategy().run(
        "t1", "a task", "system", backend, tools, AgentConfig(max_steps=3)
    )

    assert trace.final_answer == "final"
