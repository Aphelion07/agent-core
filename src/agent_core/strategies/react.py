"""Reason-and-act: one LLM call per step, re-evaluating after every
observation.

No upfront plan - the model decides the next action fresh each time, using
the full transcript including whatever tools have already returned. That
lets it recover from a bad tool result on the very next step, at the cost of
never committing to a multi-step strategy: every step is a fresh decision,
so it can also wander (re-check something it already knows, call a tool that
was never going to help) in a way a fixed plan cannot.
"""

from __future__ import annotations

from ..config import AgentConfig
from ..llm.base import LLMBackend
from ..messages import Message
from ..tools import ToolRegistry
from ..trace import RunTrace
from ._common import react_loop


class ReActStrategy:
    name = "react"

    async def run(
        self,
        task_id: str,
        task: str,
        system_prompt: str,
        backend: LLMBackend,
        tools: ToolRegistry,
        config: AgentConfig,
    ) -> RunTrace:
        trace = RunTrace(task_id=task_id, strategy=self.name)
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task),
        ]

        final = await react_loop(messages, backend, tools, config, trace, config.max_steps)
        if final is None:
            trace.hit_step_limit = True
        else:
            trace.final_answer = final
        return trace
