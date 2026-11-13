"""Measure metrics on random samples of a fixed number of tokens.

Most diversity metrics fall as more text is measured: distinct-n falls because
common n-grams repeat, compression ratio rises because the compressor finds
more repeats. Two datasets of different sizes can only be compared at the same
size. :func:`sample_to_budget` draws texts in random order, without
replacement, until exactly ``budget`` tokens are collected; the last text is
cut at the token where the budget is reached. :func:`at_budget` repeats that
draw and reports the mean of each metric with the spread across draws.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .text import Doc, total_tokens

Metric = Callable[[list[Doc]], float]


@dataclass(frozen=True)
class Estimate:
    """A metric over repeated subsamples.

    ``mean`` and ``sd`` are taken over the subsamples; ``lo`` and ``hi`` are
    the equal-tailed percentile interval of the subsample values at ``level``.
    The interval describes how much the metric moves between random samples
    of the budget size; it is not a confidence interval for the full-data
    value.
    """

    mean: float
    sd: float
    lo: float
    hi: float
    level: float
    resamples: int

    def to_dict(self) -> dict[str, float | int | None]:
        out = asdict(self)
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()}


def sample_to_budget(docs: Sequence[Doc], budget: int, rng: np.random.Generator) -> list[Doc]:
    """Draw texts in random order until exactly ``budget`` tokens are taken.

    Empty texts are never drawn. Raises ``ValueError`` when the texts hold
    fewer than ``budget`` tokens in total.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1 token")
    available = total_tokens(docs)
    if available < budget:
        raise ValueError(f"budget of {budget} tokens exceeds the {available} tokens available")
    out: list[Doc] = []
    used = 0
    for i in rng.permutation(len(docs)):
        d = docs[int(i)]
        if not d.tokens:
            continue
        need = budget - used
        if len(d) <= need:
            out.append(d)
            used += len(d)
        else:
            out.append(d.truncate(need))
            used = budget
        if used == budget:
            break
    return out


def summarize(values: Sequence[float], level: float = 0.95) -> Estimate:
    """Mean, standard deviation and percentile interval, ignoring ``nan``."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        nan = math.nan
        return Estimate(nan, nan, nan, nan, level, len(values))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(arr, [alpha, 1.0 - alpha])
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return Estimate(float(arr.mean()), sd, float(lo), float(hi), level, len(values))


def at_budget(
    docs: Sequence[Doc],
    budget: int,
    metrics: Mapping[str, Metric],
    resamples: int = 100,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Estimate]:
    """Evaluate every metric on the same ``resamples`` samples of ``budget`` tokens."""
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(resamples):
        sample = sample_to_budget(docs, budget, rng)
        for name, fn in metrics.items():
            values[name].append(fn(sample))
    return {name: summarize(v, level) for name, v in values.items()}
