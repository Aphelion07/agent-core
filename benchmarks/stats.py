from __future__ import annotations

import random
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    A 14-task run produces a noisy success rate. Without an interval,
    "reflexion scored 0.71 and react scored 0.64" reads as a result when it
    may well be sampling noise on this few tasks. This is what lets the
    README say *whether* a difference is real.
    """
    if not values:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    lower_index = int((1 - confidence) / 2 * resamples)
    upper_index = int((1 + confidence) / 2 * resamples) - 1
    return (means[lower_index], means[max(lower_index, upper_index)])
