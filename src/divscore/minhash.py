"""Near-duplicate rate with MinHash signatures and LSH banding.

Each text becomes the set of its ``k``-token shingles (a text shorter than
``k`` tokens is one shingle). A MinHash signature keeps, for each of
``num_perm`` random hash functions ``(a * x + b) mod p``, the smallest value
over the shingles; the share of positions where two signatures agree
estimates the Jaccard similarity of the two sets. Pairs that agree on all
rows of at least one band are candidates, and a candidate whose estimated
Jaccard similarity is at least ``threshold`` is a near-duplicate pair.
"""

from __future__ import annotations

import math
import zlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

PRIME = (1 << 31) - 1


def shingle_hashes(tokens: Sequence[str], k: int = 5) -> np.ndarray:
    """CRC-32 hashes (mod 2**31 - 1) of the distinct ``k``-token shingles."""
    if not tokens:
        return np.zeros(0, dtype=np.uint64)
    if len(tokens) < k:
        grams = [tuple(tokens)]
    else:
        grams = [tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]
    hashes = {zlib.crc32("\x1f".join(g).encode("utf-8")) % PRIME for g in grams}
    return np.fromiter(hashes, dtype=np.uint64, count=len(hashes))


class MinHasher:
    """A fixed family of ``num_perm`` hash functions drawn from ``seed``."""

    def __init__(self, num_perm: int = 128, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, PRIME, size=num_perm, dtype=np.uint64)

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        if hashes.size == 0:
            return np.full(self.num_perm, PRIME, dtype=np.uint64)
        vals = (np.outer(hashes, self.a) + self.b) % PRIME
        return vals.min(axis=0)


def jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity of two sets (1 for two empty sets)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicate_pairs(
    token_lists: Sequence[Sequence[str]],
    threshold: float = 0.8,
    k: int = 5,
    num_perm: int = 128,
    bands: int = 32,
    seed: int = 0,
) -> list[tuple[int, int, float]]:
    """Pairs ``(i, j, estimated_jaccard)`` at or above ``threshold``."""
    if num_perm % bands:
        raise ValueError("num_perm must be a multiple of bands")
    hasher = MinHasher(num_perm, seed)
    sigs = np.zeros((len(token_lists), num_perm), dtype=np.uint64)
    for i, t in enumerate(token_lists):
        sigs[i] = hasher.signature(shingle_hashes(t, k))
    rows = num_perm // bands
    candidates: set[tuple[int, int]] = set()
    for b in range(bands):
        buckets: defaultdict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, b * rows : (b + 1) * rows]
        for i in range(sigs.shape[0]):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    candidates.add((members[x], members[y]))
    out = []
    for i, j in sorted(candidates):
        est = float(np.mean(sigs[i] == sigs[j]))
        if est >= threshold:
            out.append((i, j, est))
    return out


def near_duplicate_rate(
    token_lists: Sequence[Sequence[str]],
    threshold: float = 0.8,
    k: int = 5,
    num_perm: int = 128,
    bands: int = 32,
    seed: int = 0,
) -> float:
    """Share of non-empty texts that have at least one near-duplicate in the set."""
    docs = [t for t in token_lists if len(t)]
    if not docs:
        return math.nan
    flagged: set[int] = set()
    for i, j, _ in near_duplicate_pairs(docs, threshold, k, num_perm, bands, seed):
        flagged.update((i, j))
    return len(flagged) / len(docs)
