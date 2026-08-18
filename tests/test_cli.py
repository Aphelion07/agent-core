"""CLI argument handling and output, with the backend swapped for the fake
so no network is touched.
"""

from __future__ import annotations

import argparse

import pytest

from agent_core import cli
from agent_core.llm.base import LLMBackend, LLMResponse
from agent_core.llm.fake import FakeBackend

_REAL_BUILD_BACKEND = cli._build_backend


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _build(args: argparse.Namespace) -> LLMBackend:
        return FakeBackend([LLMResponse(content="42")])

    monkeypatch.setattr(cli, "_build_backend", _build)


def test_runs_react_strategy_and_prints_the_answer(capsys: pytest.CaptureFixture) -> None:
    cli.main(["what is 6 times 7?", "--strategy", "react", "--max-steps", "2"])
    out = capsys.readouterr().out
    assert "strategy: react" in out
    assert "answer: 42" in out


def test_unknown_strategy_exits(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        cli.main(["a task", "--strategy", "nonexistent"])


def test_missing_task_exits(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_prints_step_limit_message_instead_of_an_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def _build(args: argparse.Namespace) -> LLMBackend:
        return FakeBackend(lambda messages, tools: LLMResponse(content=""))

    monkeypatch.setattr(cli, "_build_backend", _build)
    cli.main(["a task", "--max-steps", "0"])
    assert "hit step limit" in capsys.readouterr().out


class TestBuildBackend:
    def test_ollama_by_default(self) -> None:
        args = cli._parse_args(["a task"])
        assert _REAL_BUILD_BACKEND(args).name == "ollama"

    def test_openai_compatible_when_selected(self) -> None:
        args = cli._parse_args(["a task", "--backend", "openai_compatible", "--api-key", "sk-test"])
        assert _REAL_BUILD_BACKEND(args).name == "openai_compatible"
