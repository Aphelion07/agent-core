from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # Counts LLM calls, not tool calls - one response can request several
    # tool calls at once without spending extra budget.
    max_steps: int = 8
    context_budget_tokens: int = 4000
