"""The Vendi score with a cosine kernel on hashed n-gram counts.

Friedman and Dieng (2022) define the Vendi score of ``k`` items with a
similarity matrix ``K`` (``K[i, i] = 1``) as the exponential of the Shannon
entropy of the eigenvalues of ``K / k``. It equals 1 when all items are
identical and ``k`` when they are mutually orthogonal, and reads as an
effective number of distinct items.

Here each text is a vector of n-gram counts hashed into ``dim`` buckets and
the kernel is the cosine similarity between those vectors.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .metrics import ngrams

DEFAULT_DIM = 1 << 20


def hashed_counts(
    tokens: Sequence[str], ns: Sequence[int] = (1, 2), dim: int = DEFAULT_DIM
) -> Counter[int]:
    """Counts of the text's n-grams, hashed with CRC-32 into ``dim`` buckets."""
    out: Counter[int] = Counter()
    for n in ns:
        for g in ngrams(tokens, n):
            key = f"{n}\x1f" + "\x1f".join(g)
            out[zlib.crc32(key.encode("utf-8")) % dim] += 1
    return out


def vendi_from_kernel(k_matrix: np.ndarray) -> float:
    """Vendi score of a symmetric similarity matrix with unit diagonal."""
    k = k_matrix.shape[0]
    if k == 0:
        return math.nan
    eig = np.linalg.eigvalsh(k_matrix / k)
    eig = eig[eig > 1e-12]
    return float(math.exp(-np.sum(eig * np.log(eig))))


def cosine_kernel(vectors: Sequence[Counter[int]]) -> np.ndarray:
    """Cosine similarity matrix of sparse count vectors (none may be empty).

    Only buckets shared by at least two vectors can make an off-diagonal
    entry non-zero, so the dense product is taken over those columns alone
    and the diagonal is set to 1.
    """
    k = len(vectors)
    df: Counter[int] = Counter()
    for v in vectors:
        df.update(v.keys())
    col = {bucket: j for j, bucket in enumerate(sorted(b for b, n in df.items() if n >= 2))}
    x = np.zeros((k, len(col)), dtype=np.float64)
    for i, v in enumerate(vectors):
        norm = math.sqrt(sum(c * c for c in v.values()))
        for bucket, c in v.items():
            j = col.get(bucket)
            if j is not None:
                x[i, j] = c / norm
    kern = x @ x.T
    np.fill_diagonal(kern, 1.0)
    return kern


def ngram_vendi(
    token_lists: Sequence[Sequence[str]], ns: Sequence[int] = (1, 2), dim: int = DEFAULT_DIM
) -> float:
    """Vendi score of the texts under the hashed n-gram cosine kernel.

    Texts with no n-gram of the requested orders are left out.
    """
    vectors = [v for v in (hashed_counts(t, ns, dim) for t in token_lists) if v]
    if not vectors:
        return math.nan
    return vendi_from_kernel(cosine_kernel(vectors))
