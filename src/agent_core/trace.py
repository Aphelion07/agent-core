"""A structured record of one agent run, built for evaluation.

The benchmark needs more than pass/fail: how many LLM calls did a strategy
spend, how many were wasted on tools that turned out irrelevant, how long did
it take wall-clock. ``RunTrace`` carries all of that so ``benchmarks/bench.py``
never has to re-derive it by re-reading the transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StepKind = Literal["llm_call", "tool_call"]


@dataclass
class StepTrace:
    kind: StepKind
    detail: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    is_error: bool = False


@dataclass
class RunTrace:
    task_id: str
    strategy: str
    steps: list[StepTrace] = field(default_factory=list)
    final_answer: str = ""
    success: bool | None = None
    hit_step_limit: bool = False
    wall_time_s: float = 0.0

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "llm_call")

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool_call")

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool_call" and s.is_error)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)
