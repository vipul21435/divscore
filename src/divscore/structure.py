"""Structural repetition: openings, masked templates, type-token growth, lengths."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from .text import tokenize

WORD_RE = re.compile(r"\w")

# Quoted spans (double, curly, backtick, and single quotes not inside a word)
# and numbers are replaced by placeholders before templates are counted.
MASK_RE = re.compile(
    r"(?P<q>\"[^\"\n]{1,300}\"|“[^”\n]{1,300}”|`[^`\n]{1,300}`"
    r"|(?<!\w)'[^'\n]{1,300}'(?!\w))"
    r"|(?P<num>\d+(?:[.,:]\d+)*)"
)
QUOTE_TOKEN = "<q>"
NUMBER_TOKEN = "<num>"


def opening(tokens: Sequence[str], k: int = 3) -> tuple[str, ...] | None:
    """The first ``k`` word tokens (punctuation skipped); ``None`` if there are none."""
    words = [t for t in tokens if WORD_RE.search(t)][:k]
    return tuple(words) if words else None


def opening_counts(token_lists: Sequence[Sequence[str]], k: int = 3) -> Counter[tuple[str, ...]]:
    """How many texts start with each ``k``-word opening."""
    counts: Counter[tuple[str, ...]] = Counter()
    for tokens in token_lists:
        o = opening(tokens, k)
        if o is not None:
            counts[o] += 1
    return counts


def ranked(counts: Counter, top: int | None = None) -> list[tuple[Any, int]]:
    """Keys by count, highest first, ties broken by the key itself.

    ``Counter.most_common`` breaks ties by insertion order, which depends on
    the order of the texts and, for keys added from a set, on string hashing.
    """
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items if top is None else items[:top]


def top_share(counts: Counter, top: int = 10) -> float:
    """Share of all counted items that fall in the ``top`` most common keys."""
    total = sum(counts.values())
    if not total:
        return math.nan
    return sum(c for _, c in ranked(counts, top)) / total


def herfindahl(counts: Counter) -> float:
    """Herfindahl index: the sum of squared shares (1 when one key has everything)."""
    total = sum(counts.values())
    if not total:
        return math.nan
    return sum((c / total) ** 2 for c in counts.values())


def mask_tokens(text: str, lowercase: bool = True) -> list[str]:
    """Tokens of ``text`` with quoted spans and numbers replaced by placeholders."""
    out: list[str] = []
    pos = 0
    for m in MASK_RE.finditer(text):
        seg = text[pos : m.start()]
        out.extend(tokenize(seg, lowercase))
        out.append(QUOTE_TOKEN if m.group("q") else NUMBER_TOKEN)
        pos = m.end()
    out.extend(tokenize(text[pos:], lowercase))
    return out


def template_counts(masked: Sequence[Sequence[str]], n: int = 5) -> Counter[tuple[str, ...]]:
    """Number of texts that contain each masked ``n``-gram (document frequency)."""
    counts: Counter[tuple[str, ...]] = Counter()
    for tokens in masked:
        counts.update({tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)})
    return counts


def template_share(masked: Sequence[Sequence[str]], n: int = 5, top: int = 10) -> float:
    """Share of texts containing at least one of the ``top`` most frequent templates."""
    docs = [t for t in masked if len(t)]
    if not docs:
        return math.nan
    common = {g for g, _ in ranked(template_counts(docs, n), top)}
    hit = sum(1 for t in docs if any(tuple(t[i : i + n]) in common for i in range(len(t) - n + 1)))
    return hit / len(docs)


def type_token_curve(
    token_lists: Sequence[Sequence[str]], points: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Distinct types after the first ``N`` tokens of the texts, in the given order."""
    want = sorted({int(p) for p in points if p > 0})
    seen: set[str] = set()
    ns: list[int] = []
    vs: list[int] = []
    count = 0
    idx = 0
    for tokens in token_lists:
        for tok in tokens:
            seen.add(tok)
            count += 1
            while idx < len(want) and count == want[idx]:
                ns.append(count)
                vs.append(len(seen))
                idx += 1
        if idx == len(want):
            break
    return np.array(ns, dtype=float), np.array(vs, dtype=float)


def geometric_points(start: int, stop: int, num: int = 12) -> list[int]:
    """About ``num`` integer checkpoints spaced evenly on a log scale."""
    if stop < start:
        return [stop]
    return sorted({int(round(x)) for x in np.geomspace(start, stop, num)})


def heaps_fit(ns: np.ndarray, vs: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of Heaps' law ``V = K * N**beta`` in log space; returns (K, beta)."""
    if len(ns) < 2:
        return math.nan, math.nan
    beta, log_k = np.polyfit(np.log(ns), np.log(vs), 1)
    return float(math.exp(log_k)), float(beta)


def heaps_beta(token_lists: Sequence[Sequence[str]], start: int = 50, num: int = 12) -> float:
    """Heaps exponent fitted on the type-token curve of the texts in order."""
    total = sum(len(t) for t in token_lists)
    if total < 2 * start:
        return math.nan
    ns, vs = type_token_curve(token_lists, geometric_points(start, total, num))
    return heaps_fit(ns, vs)[1]


def length_summary(lengths: Sequence[int]) -> dict[str, float]:
    """Mean, quantiles and maximum of text lengths in tokens."""
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        return {}
    q = np.quantile(arr, [0.1, 0.5, 0.9])
    return {
        "mean": float(arr.mean()),
        "p10": float(q[0]),
        "median": float(q[1]),
        "p90": float(q[2]),
        "max": float(arr.max()),
    }


def openings_summary(
    texts: Sequence[str],
    token_lists: Sequence[Sequence[str]],
    words: int = 3,
    top: int = 20,
    template_n: int = 5,
    templates: int = 10,
    lowercase: bool = True,
) -> dict:
    """Top openings and masked templates over a whole dataset, with shares."""
    counts = opening_counts(token_lists, words)
    n = len(token_lists)
    summary: dict = {
        "n_texts": n,
        "words": words,
        "top": top,
        "top_share": top_share(counts, top),
        "herfindahl": herfindahl(counts),
        "distinct": len(counts),
        "openings": [
            {"text": " ".join(o), "count": c, "share": c / n} for o, c in ranked(counts, top)
        ],
        "template_n": template_n,
        "templates": [],
    }
    if templates > 0:
        tcounts = template_counts([mask_tokens(t, lowercase) for t in texts], template_n)
        summary["templates"] = [
            {"text": " ".join(g), "count": c, "share": c / n} for g, c in ranked(tcounts, templates)
        ]
    return summary
