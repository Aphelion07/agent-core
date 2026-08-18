from __future__ import annotations

from agent_core.tools import Tool, ToolRegistry
from conftest import EchoArgs


async def test_call_returns_output(tools: ToolRegistry) -> None:
    result = await tools.call("echo", {"text": "hi"})
    assert result.output == "echo: hi"
    assert result.error is False


async def test_call_invalid_arguments_is_an_error_not_an_exception(tools: ToolRegistry) -> None:
    result = await tools.call("echo", {})
    assert result.error is True
    assert "invalid arguments" in result.output


async def test_call_unknown_tool_lists_known_tools(tools: ToolRegistry) -> None:
    result = await tools.call("nonexistent", {})
    assert result.error is True
    assert "echo" in result.output
    assert "fail" in result.output


async def test_tool_exception_is_caught_and_reported(tools: ToolRegistry) -> None:
    result = await tools.call("fail", {})
    assert result.error is True
    assert "boom" in result.output


async def test_schema_matches_pydantic_model(echo_tool: Tool[EchoArgs]) -> None:
    schema = echo_tool.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert "text" in schema["function"]["parameters"]["properties"]


def test_registry_reports_names(tools: ToolRegistry) -> None:
    assert set(tools.names()) == {"echo", "fail"}
    assert len(tools) == 2
    assert "echo" in tools


async def test_async_tool_func_is_awaited() -> None:
    from pydantic import BaseModel

    class Args(BaseModel):
        value: int

    async def _double(args: Args) -> str:
        return str(args.value * 2)

    tool = Tool(name="double", description="double a number", parameters=Args, func=_double)
    result = await tool.run({"value": 21})
    assert result.output == "42"
