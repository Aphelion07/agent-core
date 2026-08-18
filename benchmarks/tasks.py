"""The labeled task suite the benchmark runs every strategy against.

Each task carries its own verifier instead of a single expected string,
because a correct answer can be phrased many ways ("62.1 miles", "about 62
miles", "62.14"). Verifiers check for the right number or the right kind of
admission, not an exact string match - the same reasoning
``rag-eval-lab``'s judged-vs-exact-match tension runs into, just for numbers
instead of prose.

Two tasks are deliberately unsolvable with the tools on offer (no currency
conversion, no stock data). The correct behaviour there is to say so, not to
guess - a model that always answers confidently is answering half these
tasks wrong even when the numeric-answer tasks all pass.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")
_REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "cant",
    "unable",
    "don't have",
    "do not have",
    "don't know",
    "do not know",
    "no conversion",
    "not possible",
    "no tool",
    "not available",
    "isn't available",
    "is not available",
    "no information",
    "no data",
    "not provided",
    "no way to",
    # A first pass over real model output missed most of these - "not
    # supported" and "could not be retrieved" are refusals just as much as
    # "I cannot", they just don't use the word "cannot".
    "not support",  # catches "not supported" and "does/do not support"
    "could not be",
    "couldn't be",
)


def contains_number(text: str, expected: float, tol: float = 0.6) -> bool:
    """True if any number in ``text`` is within ``tol`` of ``expected``.

    Tolerant enough to survive a model rounding differently than the
    reference calculation, tight enough that a wrong tool result or a
    dropped step still fails.
    """
    return any(abs(float(match) - expected) <= tol for match in _NUMBER_RE.findall(text))


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


@dataclass(frozen=True)
class TaskSpec:
    id: str
    prompt: str
    verify: Callable[[str], bool]
    solvable: bool = True


TASKS: list[TaskSpec] = [
    TaskSpec(
        id="calc_multi_step",
        prompt="What is (12 * 7) + (144 / 12)? Use the calculator tool, don't compute it yourself.",
        verify=lambda a: contains_number(a, 96.0),
    ),
    TaskSpec(
        id="calc_precedence",
        prompt="Compute 8 + 2 * 5 using the calculator tool, then tell me the result.",
        verify=lambda a: contains_number(a, 18.0),
    ),
    TaskSpec(
        id="unit_km_to_mi",
        prompt="Convert 100 kilometers to miles using the unit_convert tool.",
        verify=lambda a: contains_number(a, 62.1, tol=1.0),
    ),
    TaskSpec(
        id="unit_chain",
        prompt=(
            "Convert 50 kilograms to pounds using unit_convert, then add 10 to that "
            "result using the calculator. What is the final number?"
        ),
        verify=lambda a: contains_number(a, 120.2, tol=1.0),
    ),
    TaskSpec(
        id="temp_convert",
        prompt="Convert 100 degrees Celsius to Fahrenheit using the unit_convert tool.",
        verify=lambda a: contains_number(a, 212.0),
    ),
    TaskSpec(
        id="date_diff_basic",
        prompt="How many days are between 2026-01-01 and 2026-03-15? Use the date_diff tool.",
        verify=lambda a: contains_number(a, 73.0, tol=0.4),
    ),
    TaskSpec(
        id="date_and_calc",
        prompt=(
            "How many days are between 2026-06-01 and 2026-06-10? Then multiply that "
            "number by 24 to get hours, using the calculator."
        ),
        verify=lambda a: contains_number(a, 216.0, tol=1.0),
    ),
    TaskSpec(
        id="catalog_lookup_basic",
        prompt="Look up the price of SKU 'widget-42' in the catalog and tell me the price in USD.",
        verify=lambda a: contains_number(a, 19.99, tol=0.1),
    ),
    TaskSpec(
        id="catalog_then_calc",
        prompt=(
            "Look up the price of SKU 'gadget-1', then tell me the total cost for "
            "3 units using the calculator."
        ),
        verify=lambda a: contains_number(a, 387.0, tol=1.0),
    ),
    TaskSpec(
        id="unknown_sku",
        prompt=(
            "Look up the price of SKU 'does-not-exist-99' in the catalog and report "
            "exactly what the tool told you."
        ),
        verify=lambda a: "no such sku" in a.lower() or "not found" in a.lower() or is_refusal(a),
    ),
    TaskSpec(
        id="multi_tool_chain",
        prompt=(
            "Convert 26.2 miles to kilometers using unit_convert, then divide that by "
            "5 using the calculator to get the number of 5km days needed to cover it."
        ),
        verify=lambda a: contains_number(a, 8.4, tol=1.0),
    ),
    TaskSpec(
        id="distractor_check",
        prompt="What is 15 * 4? Use the calculator tool. You do not need any other tool for this.",
        verify=lambda a: contains_number(a, 60.0),
    ),
    TaskSpec(
        id="unsolvable_currency",
        prompt="Look up the price of SKU 'widget-42' and tell me its price in Japanese Yen.",
        verify=is_refusal,
        solvable=False,
    ),
    TaskSpec(
        id="unsolvable_stock",
        prompt="Look up the price of SKU 'widget-7' and tell me how many units are in stock.",
        verify=is_refusal,
        solvable=False,
    ),
]
