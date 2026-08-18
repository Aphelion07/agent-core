from __future__ import annotations

import time
from typing import Protocol

from .config import AgentConfig
from .llm.base import LLMBackend
from .tools import ToolRegistry
from .trace import RunTrace

__all__ = ["Agent", "AgentConfig", "Strategy"]


class Strategy(Protocol):
    """What a control strategy must implement.

    A strategy owns the message history and the step budget for one run; the
    ``Agent`` just times it and hands it what it needs. See
    ``agent_core.strategies`` for the three implementations.
    """

    name: str

    async def run(
        self,
        task_id: str,
        task: str,
        system_prompt: str,
        backend: LLMBackend,
        tools: ToolRegistry,
        config: AgentConfig,
    ) -> RunTrace: ...


class Agent:
    def __init__(
        self,
        backend: LLMBackend,
        tools: ToolRegistry,
        strategy: Strategy,
        config: AgentConfig | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.strategy = strategy
        self.config = config or AgentConfig()

    async def run(self, task_id: str, task: str, system_prompt: str) -> RunTrace:
        start = time.monotonic()
        trace = await self.strategy.run(
            task_id, task, system_prompt, self.backend, self.tools, self.config
        )
        trace.wall_time_s = time.monotonic() - start
        return trace
