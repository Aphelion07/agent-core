from __future__ import annotations

from agent_core.agent import Agent
from agent_core.config import AgentConfig
from agent_core.llm.base import LLMResponse
from agent_core.llm.fake import FakeBackend
from agent_core.strategies.react import ReActStrategy
from agent_core.tools import ToolRegistry


async def test_agent_run_records_wall_time(tools: ToolRegistry) -> None:
    backend = FakeBackend([LLMResponse(content="done")])
    agent = Agent(backend, tools, ReActStrategy(), AgentConfig())

    trace = await agent.run("t1", "a task", "system prompt")

    assert trace.final_answer == "done"
    assert trace.task_id == "t1"
    assert trace.strategy == "react"
    assert trace.wall_time_s >= 0.0


async def test_agent_uses_default_config_when_none_given(tools: ToolRegistry) -> None:
    backend = FakeBackend([LLMResponse(content="done")])
    agent = Agent(backend, tools, ReActStrategy())

    assert agent.config.max_steps == AgentConfig().max_steps
