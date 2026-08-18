"""Run one task through one strategy against one backend. A demo entrypoint,
not the main deliverable - see benchmarks/bench.py for the actual evaluation.
"""

from __future__ import annotations

import argparse
import asyncio

from .agent import Agent
from .config import AgentConfig
from .example_tools import calculator_tool
from .llm.base import LLMBackend
from .llm.ollama import OllamaBackend
from .llm.openai_compatible import OpenAICompatibleBackend
from .strategies import STRATEGIES
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant with access to tools. Use the calculator "
    "tool for any arithmetic instead of computing it yourself. Give a short, "
    "direct final answer once you have everything you need."
)


def _build_backend(args: argparse.Namespace) -> LLMBackend:
    if args.backend == "ollama":
        return OllamaBackend(model=args.model, base_url=args.base_url or "http://localhost:11434")
    return OpenAICompatibleBackend(
        model=args.model,
        base_url=args.base_url or "http://localhost:8000/v1",
        api_key=args.api_key,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-core")
    parser.add_argument("task", help="The task to give the agent")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="react")
    parser.add_argument("--backend", choices=["ollama", "openai_compatible"], default="ollama")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    backend = _build_backend(args)
    tools = ToolRegistry([calculator_tool()])
    strategy = STRATEGIES[args.strategy]()
    agent = Agent(backend, tools, strategy, AgentConfig(max_steps=args.max_steps))

    try:
        trace = await agent.run("cli", args.task, DEFAULT_SYSTEM_PROMPT)
    finally:
        await backend.aclose()

    print(f"strategy: {trace.strategy}")
    print(f"llm calls: {trace.llm_calls}  tool calls: {trace.tool_calls}")
    print(f"tokens: {trace.total_tokens}  wall time: {trace.wall_time_s:.2f}s")
    if trace.hit_step_limit:
        print("hit step limit without a final answer")
    else:
        print(f"answer: {trace.final_answer}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
