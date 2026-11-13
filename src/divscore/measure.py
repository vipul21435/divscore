"""The metric suites, single-dataset measurement and two-dataset comparison.

Metrics are held at one of two budgets:

- token budget: distinct-n, n-gram entropy, compression ratios and the Heaps
  exponent, measured on subsamples of exactly ``token_budget`` tokens;
- text budget: self-BLEU, ROUGE-L homogenisation, Vendi score, near-duplicate
  rate, opening concentration and template share, measured on subsamples of
  exactly ``text_budget`` whole texts. These depend on the number of texts
  compared (the Vendi score is at most the number of texts; a near-duplicate
  is found more easily among more texts), so the count is fixed instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .budget import Estimate, Metric, subsample_values, summarize
from .metrics import compression_ratio, distinct_n, ngram_entropy
from .minhash import near_duplicate_rate
from .pairwise import rouge_l_homogenization, self_bleu
from .structure import (
    heaps_beta,
    herfindahl,
    length_summary,
    mask_tokens,
    opening_counts,
    template_share,
    top_share,
)
from .text import Doc, total_tokens
from .vendi import ngram_vendi

DEFAULT_NS = (1, 2, 3)
DEFAULT_COMPRESSORS = ("gzip", "bz2")
DEFAULT_PAIRS = 500

# Direction of every metric: True when a higher value means more diverse.
DIRECTION = {
    "heaps-beta": True,
    "self-bleu": False,
    "rouge-l": False,
    "vendi": True,
    "near-dup": False,
    "opening-top10": False,
    "opening-hhi": False,
    "template-top10": False,
}
PREFIX_DIRECTION = {"distinct": True, "entropy": True, "cr": False}

# Text-budget metrics that are cheap enough to also report on the whole dataset.
WHOLE_DATA_TEXT_METRICS = ("opening-top10", "opening-hhi")


def higher_is_more_diverse(metric: str) -> bool:
    """Direction of a metric from the suites, by name."""
    if metric in DIRECTION:
        return DIRECTION[metric]
    return PREFIX_DIRECTION[metric.split("-", 1)[0]]


def _tokens(docs: list[Doc]) -> list[tuple[str, ...]]:
    return [d.tokens for d in docs]


def metric_suite(
    ns: Sequence[int] = DEFAULT_NS,
    compressors: Sequence[str] = DEFAULT_COMPRESSORS,
) -> dict[str, Metric]:
    """Token-budget metrics: distinct-n, entropy, compression ratios, Heaps exponent."""
    suite: dict[str, Metric] = {}
    for n in ns:
        suite[f"distinct-{n}"] = lambda docs, n=n: distinct_n(_tokens(docs), n)
    for n in (1, 2):
        suite[f"entropy-{n}"] = lambda docs, n=n: ngram_entropy(_tokens(docs), n)
    for method in compressors:
        suite[f"cr-{method}"] = lambda docs, m=method: compression_ratio([d.text for d in docs], m)
    suite["heaps-beta"] = lambda docs: heaps_beta(_tokens(docs))
    return suite


def text_suite(seed: int = 0, pairs: int = DEFAULT_PAIRS) -> dict[str, Metric]:
    """Text-budget metrics; ``seed`` fixes the ROUGE-L pair draws and MinHash functions."""
    pair_rng = np.random.default_rng(seed + 7919)
    return {
        "self-bleu": lambda docs: self_bleu(_tokens(docs)),
        "rouge-l": lambda docs: rouge_l_homogenization(_tokens(docs), pairs, pair_rng),
        "vendi": lambda docs: ngram_vendi(_tokens(docs)),
        "near-dup": lambda docs: near_duplicate_rate(_tokens(docs), seed=seed),
        "opening-top10": lambda docs: top_share(opening_counts(_tokens(docs)), 10),
        "opening-hhi": lambda docs: herfindahl(opening_counts(_tokens(docs))),
        "template-top10": lambda docs: template_share([mask_tokens(d.text) for d in docs]),
    }


def _clean(v: float | None) -> float | None:
    return None if v is None or math.isnan(v) else v


@dataclass
class Measurement:
    """Whole-dataset and budget-controlled values for one dataset."""

    name: str
    n_texts: int
    n_tokens: int
    token_budget: int
    text_budget: int | None
    resamples: int
    seed: int
    naive: dict[str, float | None]
    controlled: dict[str, Estimate]
    units: dict[str, str]
    lengths: dict[str, float] = field(default_factory=dict)
    skipped: int = 0
    truncate: int | None = None
    values: dict[str, list[float]] = field(default_factory=dict, repr=False)

    @property
    def mean_tokens(self) -> float:
        return self.n_tokens / self.n_texts if self.n_texts else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_texts": self.n_texts,
            "n_tokens": self.n_tokens,
            "skipped": self.skipped,
            "truncate": self.truncate,
            "lengths": self.lengths,
            "token_budget": self.token_budget,
            "text_budget": self.text_budget,
            "resamples": self.resamples,
            "seed": self.seed,
            "metrics": {
                name: {
                    "budget": self.units[name],
                    "naive": _clean(self.naive.get(name)),
                    "at_budget": est.to_dict(),
                    "higher_is_more_diverse": higher_is_more_diverse(name),
                }
                for name, est in self.controlled.items()
            },
        }


def prepare(docs: Sequence[Doc], truncate: int | None = None) -> list[Doc]:
    """Drop empty texts and cut every text to ``truncate`` tokens when given."""
    out = [d for d in docs if len(d)]
    if truncate is not None:
        if truncate < 1:
            raise ValueError("truncate must be at least 1 token")
        out = [d.truncate(truncate) for d in out]
    return out


def measure(
    docs: Sequence[Doc],
    budget: int,
    name: str = "",
    resamples: int = 100,
    seed: int = 0,
    level: float = 0.95,
    text_budget: int | None = None,
    metrics: dict[str, Metric] | None = None,
    text_metrics: dict[str, Metric] | None = None,
    skipped: int = 0,
    truncate: int | None = None,
) -> Measurement:
    """Measure one dataset at ``budget`` tokens and, if given, ``text_budget`` texts."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    pool = prepare(docs, truncate)
    tok_suite = dict(metrics) if metrics is not None else metric_suite()
    values = subsample_values(pool, budget, tok_suite, resamples, seed, unit="tokens")
    units = {k: "tokens" for k in tok_suite}
    naive: dict[str, float | None] = {k: fn(pool) for k, fn in tok_suite.items()}
    if text_budget is not None:
        txt_suite = dict(text_metrics) if text_metrics is not None else text_suite(seed)
        values.update(subsample_values(pool, text_budget, txt_suite, resamples, seed, "texts"))
        units.update({k: "texts" for k in txt_suite})
        for k, fn in txt_suite.items():
            naive[k] = fn(pool) if k in WHOLE_DATA_TEXT_METRICS else None
    return Measurement(
        name=name,
        n_texts=len(pool),
        n_tokens=total_tokens(pool),
        token_budget=budget,
        text_budget=text_budget,
        resamples=resamples,
        seed=seed,
        naive=naive,
        controlled={k: summarize(v, level) for k, v in values.items()},
        units=units,
        lengths=length_summary([len(d) for d in pool]),
        skipped=skipped + (len(docs) - len([d for d in docs if len(d)])),
        truncate=truncate,
        values=values,
    )


@dataclass
class Difference:
    """``a - b`` for one metric, with the interval of subsample differences."""

    metric: str
    unit: str
    a: float
    b: float
    diff: float
    lo: float
    hi: float
    naive_a: float | None
    naive_b: float | None

    @property
    def clear(self) -> bool:
        """The interval excludes zero."""
        return self.lo > 0 or self.hi < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "budget": self.unit,
            "a": _clean(self.a),
            "b": _clean(self.b),
            "diff": _clean(self.diff),
            "lo": _clean(self.lo),
            "hi": _clean(self.hi),
            "naive_a": _clean(self.naive_a),
            "naive_b": _clean(self.naive_b),
            "interval_excludes_zero": self.clear,
            "higher_is_more_diverse": higher_is_more_diverse(self.metric),
        }


def differences(a: Measurement, b: Measurement, level: float = 0.95) -> list[Difference]:
    """Difference of means with the percentile interval of draw-by-draw differences.

    Draw ``r`` of ``a`` and draw ``r`` of ``b`` are independent subsamples of the
    same budget, so ``a_r - b_r`` is the difference between one random sample
    of each; its spread is what the interval reports.
    """
    out = []
    alpha = (1.0 - level) / 2.0
    for name, est in a.controlled.items():
        if name not in b.controlled:
            continue
        va = np.asarray(a.values[name], dtype=float)
        vb = np.asarray(b.values[name], dtype=float)
        d = va - vb
        d = d[~np.isnan(d)]
        if d.size:
            lo, hi = (float(x) for x in np.quantile(d, [alpha, 1 - alpha]))
        else:
            lo = hi = math.nan
        out.append(
            Difference(
                metric=name,
                unit=a.units[name],
                a=est.mean,
                b=b.controlled[name].mean,
                diff=est.mean - b.controlled[name].mean,
                lo=lo,
                hi=hi,
                naive_a=a.naive.get(name),
                naive_b=b.naive.get(name),
            )
        )
    return out
