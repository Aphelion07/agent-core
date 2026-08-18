"""Backend wire-format handling, mocked at the HTTP layer with respx.

Covers request construction and response parsing for both real backends
without a running Ollama daemon or a hosted API key.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agent_core.llm.ollama import OllamaBackend
from agent_core.llm.openai_compatible import OpenAICompatibleBackend
from agent_core.messages import Message, ToolCall

OLLAMA = "http://localhost:11434"
OPENAI = "https://api.example.com/v1"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry backoff uses real asyncio.sleep - skip the wait in tests."""

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("agent_core.llm.ollama.asyncio.sleep", _instant_sleep)


@pytest.fixture
async def ollama() -> OllamaBackend:
    backend = OllamaBackend(model="qwen3:8b", base_url=OLLAMA, max_retries=2)
    yield backend
    await backend.aclose()


@pytest.fixture
async def openai_compatible() -> OpenAICompatibleBackend:
    backend = OpenAICompatibleBackend(model="gpt-test", base_url=OPENAI, api_key="test-key")
    yield backend
    await backend.aclose()


class TestOllamaBackend:
    @respx.mock
    async def test_parses_plain_text_response(self, ollama: OllamaBackend) -> None:
        respx.post(f"{OLLAMA}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "Berlin"},
                    "prompt_eval_count": 12,
                    "eval_count": 3,
                    "done": True,
                },
            )
        )
        response = await ollama.chat([Message(role="user", content="capital of Germany?")], [])
        assert response.content == "Berlin"
        assert response.prompt_tokens == 12
        assert response.completion_tokens == 3
        assert response.tool_calls == []

    @respx.mock
    async def test_parses_tool_calls_with_dict_arguments(self, ollama: OllamaBackend) -> None:
        respx.post(f"{OLLAMA}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}
                        ],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 8,
                    "done": True,
                },
            )
        )
        response = await ollama.chat(
            [Message(role="user", content="what is 1+1?")], [{"type": "function"}]
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "calculator"
        assert response.tool_calls[0].arguments == {"expression": "1+1"}

    @respx.mock
    async def test_serialises_prior_tool_calls_and_results_on_the_wire(
        self, ollama: OllamaBackend
    ) -> None:
        route = respx.post(f"{OLLAMA}/api/chat").mock(
            return_value=httpx.Response(
                200, json={"message": {"role": "assistant", "content": "done"}, "done": True}
            )
        )
        messages = [
            Message(role="user", content="add 1 and 1"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "1+1"})],
            ),
            Message(role="tool", content="2", tool_call_id="1", name="calculator"),
        ]
        await ollama.chat(messages, [])
        sent = route.calls.last.request.content
        assert b'"tool_calls"' in sent
        assert b'"role":"tool"' in sent or b'"role": "tool"' in sent

    @respx.mock
    async def test_retries_a_timeout_then_succeeds(self, ollama: OllamaBackend) -> None:
        route = respx.post(f"{OLLAMA}/api/chat")
        route.side_effect = [
            httpx.ReadTimeout("slow"),
            httpx.Response(
                200, json={"message": {"role": "assistant", "content": "ok"}, "done": True}
            ),
        ]
        response = await ollama.chat([Message(role="user", content="hi")], [])
        assert response.content == "ok"
        assert route.call_count == 2

    @respx.mock
    async def test_gives_up_after_max_retries(self, ollama: OllamaBackend) -> None:
        route = respx.post(f"{OLLAMA}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(httpx.ReadTimeout):
            await ollama.chat([Message(role="user", content="hi")], [])
        assert route.call_count == 3  # 1 initial attempt + 2 retries


class TestOpenAICompatibleBackend:
    @respx.mock
    async def test_parses_response_and_usage(
        self, openai_compatible: OpenAICompatibleBackend
    ) -> None:
        respx.post(f"{OPENAI}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "Paris"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )
        )
        response = await openai_compatible.chat(
            [Message(role="user", content="capital of France?")], []
        )
        assert response.content == "Paris"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 2

    @respx.mock
    async def test_parses_tool_calls_with_json_string_arguments(
        self, openai_compatible: OpenAICompatibleBackend
    ) -> None:
        respx.post(f"{OPENAI}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "calculator",
                                            "arguments": '{"expression": "2+2"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        )
        response = await openai_compatible.chat([Message(role="user", content="what is 2+2?")], [])
        assert response.content == ""
        assert response.tool_calls[0].id == "call_1"
        assert response.tool_calls[0].arguments == {"expression": "2+2"}

    @respx.mock
    async def test_sends_bearer_token(self, openai_compatible: OpenAICompatibleBackend) -> None:
        route = respx.post(f"{OPENAI}/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            )
        )
        await openai_compatible.chat([Message(role="user", content="hi")], [])
        assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"

    @respx.mock
    async def test_malformed_tool_arguments_do_not_crash(
        self, openai_compatible: OpenAICompatibleBackend
    ) -> None:
        respx.post(f"{OPENAI}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "calculator", "arguments": "not json"},
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        )
        response = await openai_compatible.chat([Message(role="user", content="hi")], [])
        assert response.tool_calls[0].arguments == {}
