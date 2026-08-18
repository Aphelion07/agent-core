"""Backend for any OpenAI-compatible ``/chat/completions`` endpoint.

Points at a hosted API, a local server such as vLLM, or - usefully for this
portfolio - at ``llm-gateway`` from the sibling project, which adds caching
and provider failover in front of whatever it proxies to. Tool-call arguments
arrive as a JSON string on this wire format, unlike the Ollama native API.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..messages import Message, ToolCall
from .base import LLMBackend, LLMResponse


class OpenAICompatibleBackend(LLMBackend):
    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    def _to_wire(self, message: Message) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in message.tool_calls
            ]
        if message.role == "tool" and message.tool_call_id:
            wire["tool_call_id"] = message.tool_call_id
        return wire

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_wire(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}

        tool_calls = []
        for tc in message.get("tool_calls") or []:
            function = tc.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=function.get("name", ""), arguments=arguments)
            )

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
