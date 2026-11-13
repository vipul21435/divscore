"""Corpus-level lexical diversity metrics.

Each function takes a list of token sequences (one per text) or a list of raw
texts and returns a single number. None of them corrects for corpus size on
its own; :mod:`divscore.budget` does that by measuring every metric on samples
of the same number of tokens.

N-grams never cross a text boundary.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence

COMPRESSORS = ("gzip", "bz2", "lzma")


def ngrams(tokens: Sequence[str], n: int) -> Iterator[tuple[str, ...]]:
    """Yield the contiguous n-grams of one token sequence."""
    if n < 1:
        raise ValueError("n must be at least 1")
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def ngram_counts(token_lists: Iterable[Sequence[str]], n: int) -> Counter[tuple[str, ...]]:
    """Count n-grams over all texts."""
    counts: Counter[tuple[str, ...]] = Counter()
    for tokens in token_lists:
        counts.update(ngrams(tokens, n))
    return counts


def distinct_n(token_lists: Iterable[Sequence[str]], n: int, per: str = "ngram") -> float:
    """Number of distinct n-grams divided by a count of the text.

    ``per="ngram"`` (default) divides by the total number of n-grams, so the
    value lies in (0, 1]. ``per="token"`` divides by the total number of
    tokens, which is how Li et al. (2016) state distinct-1 and distinct-2
    ("divided by total number of generated words"). The two agree for
    ``n = 1``. Returns ``nan`` when the texts hold no n-gram of that order.
    """
    if per not in ("ngram", "token"):
        raise ValueError("per must be 'ngram' or 'token'")
    lists = list(token_lists)
    counts = ngram_counts(lists, n)
    total = sum(counts.values())
    if not total:
        return math.nan
    if per == "token":
        total = sum(len(t) for t in lists)
    return len(counts) / total


def ngram_entropy(token_lists: Iterable[Sequence[str]], n: int) -> float:
    """Shannon entropy, in bits, of the empirical n-gram distribution."""
    counts = ngram_counts(token_lists, n)
    total = sum(counts.values())
    if not total:
        return math.nan
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def _compress(data: bytes, method: str) -> bytes:
    if method == "gzip":
        return gzip.compress(data, compresslevel=9, mtime=0)
    if method == "bz2":
        return bz2.compress(data, compresslevel=9)
    if method == "lzma":
        return lzma.compress(data, preset=9)
    raise ValueError(f"unknown compressor {method!r}; expected one of {', '.join(COMPRESSORS)}")


def compression_ratio(texts: Iterable[str], method: str = "gzip") -> float:
    """Uncompressed size over compressed size of the texts joined by newlines.

    Higher means more repetition, so lower is more diverse. The fixed header
    of each format is included, which matters for very small inputs; compare
    values only at the same token budget.
    """
    data = "\n".join(texts).encode("utf-8")
    if not data:
        return math.nan
    return len(data) / len(_compress(data, method))
