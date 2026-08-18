"""Tool definitions and the registry an agent chooses from.

A tool pairs a pydantic model (its argument schema) with a plain function.
The schema doubles as the JSON schema advertised to the model, so there is
only one place that can drift: write the pydantic model, the wire format and
the validation follow for free.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ToolResult:
    output: str
    error: bool = False


ModelT = TypeVar("ModelT", bound=BaseModel)
ToolFunc = Callable[[ModelT], str] | Callable[[ModelT], Awaitable[str]]


class Tool(Generic[ModelT]):
    """Generic over its argument model so a tool's ``func`` can take the
    concrete pydantic subclass rather than bare ``BaseModel`` - the whole
    point of giving each tool its own schema.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: type[ModelT],
        func: ToolFunc[ModelT],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function schema. Ollama and OpenAI-compatible
        endpoints both accept this shape, so one schema serves both backends.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            args = self.parameters.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(output=f"invalid arguments for {self.name}: {exc}", error=True)

        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(args)
            else:
                result = await asyncio.to_thread(self.func, args)
        except Exception as exc:
            # A bug in a tool's own implementation must not crash the agent
            # loop - it becomes an observation the model can react to, same
            # as a bad lookup or a validation failure.
            return ToolResult(output=f"{self.name} raised {type(exc).__name__}: {exc}", error=True)

        return ToolResult(output=str(result))


class ToolRegistry:
    def __init__(self, tools: list[Tool[Any]] | None = None) -> None:
        self._tools: dict[str, Tool[Any]] = {t.name: t for t in (tools or [])}

    def register(self, tool: Tool[Any]) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(self.names()) or "(none)"
            return ToolResult(output=f"unknown tool '{name}'. Known tools: {known}", error=True)
        return await tool.run(arguments)
