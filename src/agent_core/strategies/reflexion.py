"""ReAct plus one self-critique round before committing to an answer.

Runs the same loop as ReAct but reserves the last call in the step budget
for a verification pass: the model is shown its own candidate answer and
asked to confirm or correct it. That catches some mistakes ReAct would have
shipped outright, but it is not free - the reserved call is one fewer step
available for actual tool use, so on tasks that need every step of the
budget for legwork, reflexion can run out of room where plain ReAct would
not.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from ..config import AgentConfig
from ..llm.base import LLMBackend
from ..messages import Message
from ..tools import Tool, ToolRegistry
from ..trace import RunTrace
from ._common import call_llm, react_loop


class VerdictArgs(BaseModel):
    correct: bool
    corrected_answer: str | None = None


def _verdict_tool() -> Tool[VerdictArgs]:
    return Tool(
        name="submit_verdict",
        description=(
            "Record whether the candidate answer is correct. If it is not, "
            "provide corrected_answer with the right answer instead."
        ),
        parameters=VerdictArgs,
        func=lambda _args: "verdict recorded",
    )


class ReflexionStrategy:
    name = "reflexion"

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

        react_budget = max(1, config.max_steps - 1)
        candidate = await react_loop(messages, backend, tools, config, trace, react_budget)
        if candidate is None:
            trace.hit_step_limit = True
            return trace
        trace.final_answer = candidate

        if config.max_steps - react_budget < 1:
            return trace  # no budget left for a critique round

        critique_messages = [
            *messages,
            Message(role="assistant", content=candidate),
            Message(
                role="user",
                content=(
                    f"Double-check your answer above against the original task ({task!r}). "
                    "Call submit_verdict with correct=true if it is right, or correct=false "
                    "plus a corrected_answer if it is wrong."
                ),
            ),
        ]
        response = await call_llm(
            backend,
            critique_messages,
            [_verdict_tool().schema()],
            config.context_budget_tokens,
            trace,
        )
        verdict_call = next((c for c in response.tool_calls if c.name == "submit_verdict"), None)
        if verdict_call is None:
            return trace

        try:
            verdict = VerdictArgs.model_validate(verdict_call.arguments)
        except ValidationError:
            return trace

        if not verdict.correct and verdict.corrected_answer:
            trace.final_answer = verdict.corrected_answer
        return trace
