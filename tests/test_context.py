from __future__ import annotations

from agent_core.context import trim_to_budget
from agent_core.messages import Message, Role


def _msg(role: Role, chars: int) -> Message:
    return Message(role=role, content="x" * chars)


def test_keeps_everything_when_under_budget() -> None:
    messages = [_msg("system", 40), _msg("user", 40), _msg("assistant", 40)]
    assert trim_to_budget(messages, max_tokens=1000) == messages


def test_system_messages_are_never_dropped() -> None:
    messages = [_msg("system", 4000)] + [_msg("user", 400) for _ in range(20)]
    trimmed = trim_to_budget(messages, max_tokens=1100)
    assert trimmed[0].role == "system"
    assert sum(1 for m in trimmed if m.role == "system") == 1


def test_drops_oldest_non_system_messages_first() -> None:
    # 46-char bodies -> 11 approx-tokens each, so only one fits alongside a
    # 10-token system message inside a 25-token budget.
    messages = [
        _msg("system", 40),
        Message(role="user", content="oldest" + "-" * 40),
        Message(role="user", content="middle" + "-" * 40),
        Message(role="user", content="newest" + "-" * 40),
    ]
    trimmed = trim_to_budget(messages, max_tokens=25)
    contents = [m.content for m in trimmed]
    assert any("newest" in c for c in contents)
    assert not any("oldest" in c for c in contents)


def test_always_keeps_at_least_the_most_recent_message() -> None:
    messages = [_msg("system", 10), _msg("user", 5000)]
    trimmed = trim_to_budget(messages, max_tokens=1)
    assert trimmed[-1].role == "user"


def test_empty_messages_returns_empty() -> None:
    assert trim_to_budget([], max_tokens=100) == []
