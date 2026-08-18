"""Measure which agent control strategy actually works, instead of picking
one and shipping it.

Runs all three strategies (react, plan_execute, reflexion) against the same
14-task suite and reports success rate with a bootstrap confidence interval,
plus the cost side of the ledger: LLM calls, tool calls, tokens, wall time.
Two tasks are unsolvable on purpose - the tools to answer them don't exist -
so "success" also covers whether a strategy admits that instead of guessing.

Defaults run against a local Ollama daemon:

    python benchmarks/bench.py --model qwen3:8b --runs 3

The ``fake`` backend needs no model and no network - it answers every
planning/verdict tool call correctly and every task tool call with a
placeholder, which exercises every code path without asserting on accuracy.
That is what CI's smoke job runs, to catch a broken strategy or a crashing
task before it reaches a real benchmark:

    python benchmarks/bench.py --backend fake --runs 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import bootstrap_ci, mean
from tasks import TASKS, TaskSpec
from tools import build_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_core.agent import Agent
from agent_core.config import AgentConfig
from agent_core.llm.base import LLMBackend, LLMResponse
from agent_core.llm.fake import FakeBackend
from agent_core.llm.ollama import OllamaBackend
from agent_core.llm.openai_compatible import OpenAICompatibleBackend
from agent_core.messages import Message, ToolCall
from agent_core.strategies import STRATEGIES
from agent_core.trace import RunTrace

SYSTEM_PROMPT = (
    "You are a careful assistant with access to tools: calculator, "
    "catalog_lookup, unit_convert, date_diff and get_weather. Use a tool "
    "whenever a task needs one instead of computing or recalling the answer "
    "yourself. If the task cannot be solved with the tools available, say so "
    "plainly instead of guessing. Give a short, direct final answer."
)


def _smoke_fake_script(messages: list[Message], tool_schemas: list[dict[str, Any]]) -> LLMResponse:
    """Scripted answers for ``--backend fake``: correct on the synthetic
    control tools (submit_plan, submit_verdict) so plan_execute and
    reflexion run their full control flow, and a placeholder everywhere
    else. Never asserted against task verifiers - this backend measures
    whether the harness runs, not whether it is smart.
    """
    names = {t["function"]["name"] for t in tool_schemas}
    if "submit_plan" in names:
        return LLMResponse(
            tool_calls=[ToolCall(id="p", name="submit_plan", arguments={"steps": ["solve it"]})]
        )
    if "submit_verdict" in names:
        return LLMResponse(
            tool_calls=[ToolCall(id="v", name="submit_verdict", arguments={"correct": True})]
        )
    return LLMResponse(content="0")


def _build_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "fake":
        return FakeBackend(_smoke_fake_script)
    if args.backend == "ollama":
        return OllamaBackend(model=args.model, base_url=args.base_url or "http://localhost:11434")
    return OpenAICompatibleBackend(
        model=args.model, base_url=args.base_url or "http://localhost:8000/v1", api_key=args.api_key
    )


def _called_distractor(trace: RunTrace) -> bool:
    return any(s.kind == "tool_call" and s.detail.startswith("get_weather(") for s in trace.steps)


async def _run_one(
    backend: LLMBackend, strategy_name: str, task: TaskSpec, config: AgentConfig
) -> RunTrace:
    strategy = STRATEGIES[strategy_name]()
    agent = Agent(backend, build_registry(), strategy, config)
    trace = await agent.run(task.id, task.prompt, SYSTEM_PROMPT)
    trace.success = (not trace.hit_step_limit) and task.verify(trace.final_answer)
    return trace


async def run_matrix(
    args: argparse.Namespace, on_progress: Callable[[list[RunTrace]], None] | None = None
) -> list[RunTrace]:
    """Runs the full strategy x task x repeat matrix.

    ``on_progress`` fires after every completed run with the traces
    collected so far, so the caller can checkpoint to disk - a run against a
    real model takes long enough that one transient network error losing
    everything is not acceptable.
    """
    strategy_names = args.strategies or sorted(STRATEGIES)
    tasks = [t for t in TASKS if not args.tasks or t.id in args.tasks]
    config = AgentConfig(max_steps=args.max_steps)

    backend = _build_backend(args)
    traces: list[RunTrace] = []
    try:
        for strategy_name in strategy_names:
            for task in tasks:
                for run_idx in range(args.runs):
                    trace = await _run_one(backend, strategy_name, task, config)
                    traces.append(trace)
                    status = (
                        "ok" if trace.success else ("limit" if trace.hit_step_limit else "fail")
                    )
                    print(
                        f"[{strategy_name}] {task.id} run {run_idx + 1}/{args.runs}: "
                        f"{status}  calls={trace.llm_calls} tokens={trace.total_tokens} "
                        f"({trace.wall_time_s:.1f}s)"
                    )
                    if on_progress is not None:
                        on_progress(traces)
    finally:
        await backend.aclose()
    return traces


def summarize(traces: list[RunTrace]) -> dict[str, Any]:
    task_by_id = {t.id: t for t in TASKS}
    by_strategy: dict[str, list[RunTrace]] = defaultdict(list)
    for trace in traces:
        by_strategy[trace.strategy].append(trace)

    summary: dict[str, Any] = {}
    for strategy_name, strategy_traces in by_strategy.items():
        solvable = [t for t in strategy_traces if task_by_id[t.task_id].solvable]
        unsolvable = [t for t in strategy_traces if not task_by_id[t.task_id].solvable]
        successes = [1.0 if t.success else 0.0 for t in strategy_traces]
        lo, hi = bootstrap_ci(successes)

        summary[strategy_name] = {
            "runs": len(strategy_traces),
            "success_rate": mean(successes),
            "success_rate_ci95": [lo, hi],
            "solvable_success_rate": mean([1.0 if t.success else 0.0 for t in solvable]),
            "unsolvable_refusal_rate": mean([1.0 if t.success else 0.0 for t in unsolvable]),
            "hit_step_limit_rate": mean(
                [1.0 if t.hit_step_limit else 0.0 for t in strategy_traces]
            ),
            "distractor_call_rate": mean(
                [1.0 if _called_distractor(t) else 0.0 for t in strategy_traces]
            ),
            "mean_llm_calls": mean([float(t.llm_calls) for t in strategy_traces]),
            "mean_tool_calls": mean([float(t.tool_calls) for t in strategy_traces]),
            "mean_total_tokens": mean([float(t.total_tokens) for t in strategy_traces]),
            "mean_wall_time_s": mean([t.wall_time_s for t in strategy_traces]),
        }
    return summary


def per_task_breakdown(traces: list[RunTrace]) -> dict[str, float]:
    by: dict[tuple[str, str], list[RunTrace]] = defaultdict(list)
    for trace in traces:
        by[(trace.strategy, trace.task_id)].append(trace)
    return {
        f"{strategy_name}/{task_id}": mean([1.0 if t.success else 0.0 for t in ts])
        for (strategy_name, task_id), ts in sorted(by.items())
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend", choices=["fake", "ollama", "openai_compatible"], default="ollama"
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--strategies", nargs="*", default=None, choices=sorted(STRATEGIES))
    parser.add_argument("--tasks", nargs="*", default=None, choices=[t.id for t in TASKS])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", default="benchmarks/results.json")
    return parser.parse_args(argv)


def build_result(args: argparse.Namespace, traces: list[RunTrace]) -> dict[str, Any]:
    return {
        "backend": args.backend,
        "model": args.model,
        "max_steps": args.max_steps,
        "runs_per_task": args.runs,
        "summary": summarize(traces),
        "per_task_success_rate": per_task_breakdown(traces),
        "traces": [
            {
                "task_id": t.task_id,
                "strategy": t.strategy,
                "success": t.success,
                "hit_step_limit": t.hit_step_limit,
                "llm_calls": t.llm_calls,
                "tool_calls": t.tool_calls,
                "total_tokens": t.total_tokens,
                "wall_time_s": t.wall_time_s,
                "final_answer": t.final_answer,
                "steps": [asdict(s) for s in t.steps],
            }
            for t in traces
        ],
    }


async def _main(argv: list[str] | None) -> None:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(traces: list[RunTrace]) -> None:
        out_path.write_text(json.dumps(build_result(args, traces), indent=2))

    traces = await run_matrix(args, on_progress=checkpoint)
    result = build_result(args, traces)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n=== summary ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_main(argv))


if __name__ == "__main__":
    main()
