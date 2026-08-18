from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_core.tools import Tool, ToolRegistry


class EchoArgs(BaseModel):
    text: str


class FailArgs(BaseModel):
    pass


def _echo(args: EchoArgs) -> str:
    return f"echo: {args.text}"


def _always_fail(args: FailArgs) -> str:
    raise RuntimeError("boom")


@pytest.fixture
def echo_tool() -> Tool[EchoArgs]:
    return Tool(
        name="echo", description="Echo the given text back.", parameters=EchoArgs, func=_echo
    )


@pytest.fixture
def failing_tool() -> Tool[FailArgs]:
    return Tool(name="fail", description="Always raises.", parameters=FailArgs, func=_always_fail)


@pytest.fixture
def tools(echo_tool: Tool[EchoArgs], failing_tool: Tool[FailArgs]) -> ToolRegistry:
    return ToolRegistry([echo_tool, failing_tool])
