"""Ollama backend - the default, and the reason this project needs no API key.

Talks to the daemon's native ``/api/chat`` endpoint. Unlike the OpenAI shim,
the native route reports real token counts (``prompt_eval_count`` /
``eval_count``) and returns tool-call arguments as a parsed object rather
than a JSON string, which is what ``arguments_to_dict`` below normalises for.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from ..messages import Message, ToolCall
from .base import LLMBackend, LLMResponse


def _arguments_to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    def _to_wire(self, message: Message) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            wire["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in message.tool_calls
            ]
        return wire

    async def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        """A local daemon can still stall - a cold model load, a busy GPU
        queue - long enough to trip the read timeout without anything being
        wrong. One or two retries absorb that without masking a real outage,
        which would keep failing past ``max_retries``.
        """
        last_error: httpx.TransportError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_wire(m) for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        response = await self._post_with_retries(payload)
        data = response.json()
        message = data.get("message", {})

        tool_calls = [
            ToolCall(
                id=f"call_{i}",
                name=tc.get("function", {}).get("name", ""),
                arguments=_arguments_to_dict(tc.get("function", {}).get("arguments")),
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]

        return LLMResponse(
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
