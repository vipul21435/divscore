"""The default metric suite and a single-dataset measurement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .budget import Estimate, Metric, at_budget
from .metrics import compression_ratio, distinct_n, ngram_entropy
from .text import Doc, total_tokens

DEFAULT_NS = (1, 2, 3)
DEFAULT_COMPRESSORS = ("gzip", "bz2")

# For each metric family: does a higher value mean more diverse?
HIGHER_IS_MORE_DIVERSE = {"distinct": True, "entropy": True, "cr": False}


def _distinct(n: int) -> Metric:
    return lambda docs: distinct_n([d.tokens for d in docs], n)


def _entropy(n: int) -> Metric:
    return lambda docs: ngram_entropy([d.tokens for d in docs], n)


def _compression(method: str) -> Metric:
    return lambda docs: compression_ratio([d.text for d in docs], method)


def metric_suite(
    ns: Sequence[int] = DEFAULT_NS,
    compressors: Sequence[str] = DEFAULT_COMPRESSORS,
) -> dict[str, Metric]:
    """Distinct-n for each ``n``, unigram and bigram entropy, compression ratios."""
    suite: dict[str, Metric] = {}
    for n in ns:
        suite[f"distinct-{n}"] = _distinct(n)
    for n in (1, 2):
        suite[f"entropy-{n}"] = _entropy(n)
    for method in compressors:
        suite[f"cr-{method}"] = _compression(method)
    return suite


def higher_is_more_diverse(metric: str) -> bool:
    """Direction of a metric from the suite, by its name prefix."""
    return HIGHER_IS_MORE_DIVERSE[metric.split("-", 1)[0]]


def _clean(v: float) -> float | None:
    return None if math.isnan(v) else v


@dataclass
class Measurement:
    """Naive (whole dataset) and budget-controlled values for one dataset."""

    name: str
    n_texts: int
    n_tokens: int
    budget: int
    resamples: int
    seed: int
    naive: dict[str, float]
    controlled: dict[str, Estimate]
    skipped: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_tokens(self) -> float:
        return self.n_tokens / self.n_texts if self.n_texts else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_texts": self.n_texts,
            "n_tokens": self.n_tokens,
            "mean_tokens": _clean(self.mean_tokens),
            "skipped": self.skipped,
            "budget": self.budget,
            "resamples": self.resamples,
            "seed": self.seed,
            "metrics": {
                name: {
                    "naive": _clean(self.naive[name]),
                    "at_budget": self.controlled[name].to_dict(),
                    "higher_is_more_diverse": higher_is_more_diverse(name),
                }
                for name in self.controlled
            },
            **self.extra,
        }


def measure(
    docs: Sequence[Doc],
    budget: int,
    name: str = "",
    resamples: int = 100,
    seed: int = 0,
    level: float = 0.95,
    metrics: Mapping[str, Metric] | None = None,
    skipped: int = 0,
) -> Measurement:
    """Measure the suite on the whole dataset and at ``budget`` tokens."""
    suite = dict(metrics) if metrics is not None else metric_suite()
    nonempty = [d for d in docs if len(d)]
    naive = {k: fn(nonempty) for k, fn in suite.items()}
    controlled = at_budget(nonempty, budget, suite, resamples=resamples, seed=seed, level=level)
    return Measurement(
        name=name,
        n_texts=len(nonempty),
        n_tokens=total_tokens(nonempty),
        budget=budget,
        resamples=resamples,
        seed=seed,
        naive=naive,
        controlled=controlled,
        skipped=skipped + (len(docs) - len(nonempty)),
    )
