from __future__ import annotations

import pytest

from agent_core.example_tools import calculator_tool


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 1", "2.0"),
        ("(12 * 7) + (144 / 12)", "96.0"),
        ("2 ** 10", "1024.0"),
        ("-5 + 3", "-2.0"),
    ],
)
async def test_calculator_evaluates_arithmetic(expression: str, expected: str) -> None:
    result = await calculator_tool().run({"expression": expression})
    assert result.output == expected
    assert result.error is False


async def test_calculator_rejects_unsafe_expressions() -> None:
    result = await calculator_tool().run({"expression": "__import__('os').system('echo hi')"})
    assert result.error is True
