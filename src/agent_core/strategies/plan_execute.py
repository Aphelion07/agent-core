"""Plan first, execute without re-planning.

One call produces an ordered list of steps up front; each step then gets
exactly one execution turn, in order, regardless of what earlier steps
returned. This is the mirror image of ReAct's weakness: a good plan runs
cheaper (no re-deciding what to do at every turn) and stays on-track for
long sequences, but a plan built on a wrong assumption executes to
completion anyway - there is no step here that reconsiders step one after
step three's tool result contradicts it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from ..config import AgentConfig
from ..llm.base import LLMBackend
from ..messages import Message
from ..tools import Tool, ToolRegistry
from ..trace import RunTrace
from ._common import call_llm, execute_tool_calls


class PlanArgs(BaseModel):
    steps: list[str] = Field(min_length=1)


def _plan_tool() -> Tool[PlanArgs]:
    return Tool(
        name="submit_plan",
        description="Submit an ordered list of concrete steps to execute, in order, to solve the "
        "task.",
        parameters=PlanArgs,
        func=lambda _args: "plan recorded",
    )


class PlanExecuteStrategy:
    name = "plan_execute"

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

        plan_request = Message(
            role="user",
            content="Break the task above into an ordered list of concrete steps, then call "
            "submit_plan.",
        )
        plan_response = await call_llm(
            backend,
            [*messages, plan_request],
            [_plan_tool().schema()],
            config.context_budget_tokens,
            trace,
        )

        plan_call = next((c for c in plan_response.tool_calls if c.name == "submit_plan"), None)
        plan: list[str]
        if plan_call is not None:
            try:
                plan = PlanArgs.model_validate(plan_call.arguments).steps
            except ValidationError:
                plan = [task]
        else:
            # The model answered in plain text instead of calling
            # submit_plan - degrade to a single-step plan rather than
            # failing the run outright.
            plan = [task]

        remaining = config.max_steps - 1  # the planning call already spent one
        if remaining <= 0:
            trace.final_answer = plan_response.content
            trace.hit_step_limit = not trace.final_answer
            return trace

        exec_budget = max(0, remaining - 1)  # reserve one call for the final answer
        for step in plan[:exec_budget]:
            messages.append(Message(role="user", content=f"Step: {step}"))
            step_response = await call_llm(
                backend, messages, tools.schemas(), config.context_budget_tokens, trace
            )
            if step_response.tool_calls:
                messages.extend(await execute_tool_calls(step_response, tools, trace))
            else:
                messages.append(Message(role="assistant", content=step_response.content))

        messages.append(
            Message(role="user", content=f"All steps are done. Give the final answer to: {task}")
        )
        final_response = await call_llm(backend, messages, [], config.context_budget_tokens, trace)
        trace.final_answer = final_response.content
        return trace
