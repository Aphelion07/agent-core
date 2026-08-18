"""Building blocks shared by the three strategies.

Each strategy composes these differently: ReAct calls ``react_loop`` once
with its full step budget, Reflexion calls it with a smaller budget and
spends the rest on a critique round, Plan-and-Execute skips it entirely in
favour of one call per pre-committed step.
"""

from __future__ import annotations

from typing import Any

from ..config import AgentConfig
from ..context import trim_to_budget
from ..llm.base import LLMBackend, LLMResponse
from ..messages import Message
from ..tools import ToolRegistry
from ..trace import RunTrace, StepTrace


async def call_llm(
    backend: LLMBackend,
    messages: list[Message],
    tool_schemas: list[dict[str, Any]],
    context_budget_tokens: int,
    trace: RunTrace,
) -> LLMResponse:
    trimmed = trim_to_budget(messages, context_budget_tokens)
    response = await backend.chat(trimmed, tool_schemas)
    detail = (
        response.content[:200] if response.content else f"{len(response.tool_calls)} tool call(s)"
    )
    trace.steps.append(
        StepTrace(
            kind="llm_call",
            detail=detail,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
    )
    return response


async def execute_tool_calls(
    response: LLMResponse, tools: ToolRegistry, trace: RunTrace
) -> list[Message]:
    observations: list[Message] = [
        Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
    ]
    for call in response.tool_calls:
        result = await tools.call(call.name, call.arguments)
        trace.steps.append(
            StepTrace(
                kind="tool_call",
                detail=f"{call.name}({call.arguments}) -> {result.output[:200]}",
                is_error=result.error,
            )
        )
        observations.append(
            Message(role="tool", content=result.output, tool_call_id=call.id, name=call.name)
        )
    return observations


async def react_loop(
    messages: list[Message],
    backend: LLMBackend,
    tools: ToolRegistry,
    config: AgentConfig,
    trace: RunTrace,
    budget_steps: int,
) -> str | None:
    """Runs ReAct-style turns against ``messages`` (mutated in place) until
    the model stops requesting tools or the budget runs out. Returns the
    final free-text answer, or ``None`` if the budget was exhausted first.
    """
    for _ in range(budget_steps):
        response = await call_llm(
            backend, messages, tools.schemas(), config.context_budget_tokens, trace
        )
        if not response.tool_calls:
            return response.content
        messages.extend(await execute_tool_calls(response, tools, trace))
    return None
