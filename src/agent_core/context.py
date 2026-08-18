"""Token-budget trimming for the running conversation.

Long tool-use runs accumulate observations fast - a handful of steps with
verbose tool output can blow past a small model's context window well before
the step budget is reached. Trimming keeps every system message (the task
framing the model needs throughout) and as much of the recent tail as fits,
dropping the oldest non-system messages first.
"""

from __future__ import annotations

from .messages import Message


def trim_to_budget(messages: list[Message], max_tokens: int) -> list[Message]:
    if not messages:
        return messages

    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]

    budget = max_tokens - sum(m.approx_tokens() for m in system)
    kept: list[Message] = []
    for message in reversed(rest):
        cost = message.approx_tokens()
        # Always keep at least the single most recent message, even if it
        # alone exceeds the remaining budget - an empty tail would leave the
        # model with no observation to react to.
        if cost > budget and kept:
            break
        budget -= cost
        kept.append(message)
    kept.reverse()

    return system + kept
