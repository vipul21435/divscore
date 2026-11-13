"""Similarity between the texts of a sample: self-BLEU and ROUGE-L homogenisation.

Self-BLEU follows the Texygen definition (Zhu et al., 2018): each text is a
hypothesis scored by sentence BLEU against all other texts of the sample as
references, and the scores are averaged. BLEU uses uniform weights up to
``max_n``, clipped counts against the largest count in any one reference, the
closest reference length for the brevity penalty, and the "method 1"
smoothing of NLTK (a numerator of 0.1 for an order with no match), so that a
single text matches ``nltk.translate.bleu_score.sentence_bleu`` with
``SmoothingFunction().method1``.

Shaib et al. (2024) define homogenisation as the mean ROUGE-L similarity
over all pairs of texts. Here the similarity is ROUGE-L F1 and, when there
are more pairs than requested, the pairs are drawn at random with
replacement (each pair joins two different texts).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .metrics import ngrams

EPSILON = 0.1  # NLTK SmoothingFunction().method1 default


def _brevity_penalty(hyp_len: int, ref_lens: Sequence[int]) -> float:
    if hyp_len == 0:
        return 0.0
    closest = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
    if hyp_len > closest:
        return 1.0
    return math.exp(1.0 - closest / hyp_len)


def _bleu_from_counts(
    matches: Sequence[int], totals: Sequence[int], hyp_len: int, ref_lens: Sequence[int]
) -> float:
    if not matches or matches[0] == 0:
        return 0.0
    w = 1.0 / len(matches)
    log_sum = 0.0
    for m, t in zip(matches, totals, strict=True):
        denom = max(1, t)
        p = m / denom if m > 0 else EPSILON / denom
        log_sum += w * math.log(p)
    return _brevity_penalty(hyp_len, ref_lens) * math.exp(log_sum)


def sentence_bleu(hyp: Sequence[str], refs: Sequence[Sequence[str]], max_n: int = 4) -> float:
    """BLEU of one hypothesis against several references."""
    if not refs:
        raise ValueError("at least one reference is needed")
    matches: list[int] = []
    totals: list[int] = []
    for n in range(1, max_n + 1):
        hc = Counter(ngrams(hyp, n))
        best: Counter[tuple[str, ...]] = Counter()
        for r in refs:
            for g, c in Counter(ngrams(r, n)).items():
                if c > best[g]:
                    best[g] = c
        matches.append(sum(min(c, best[g]) for g, c in hc.items()))
        totals.append(sum(hc.values()))
    return _bleu_from_counts(matches, totals, len(hyp), [len(r) for r in refs])


def self_bleu(token_lists: Sequence[Sequence[str]], max_n: int = 4) -> float:
    """Mean BLEU of each text against all the others as references.

    Runs in time linear in the number of n-grams: for every n-gram it keeps
    the largest and second-largest count over the texts, which gives the
    clipping count with any one text left out.
    """
    docs = [list(t) for t in token_lists if len(t)]
    k = len(docs)
    if k < 2:
        return math.nan
    per_order: list[tuple[list[Counter], dict]] = []
    for n in range(1, max_n + 1):
        counters = [Counter(ngrams(d, n)) for d in docs]
        top: dict[tuple[str, ...], list[int]] = {}
        for i, c in enumerate(counters):
            for g, v in c.items():
                t = top.get(g)
                if t is None:
                    top[g] = [v, i, 0]
                elif v > t[0]:
                    t[2], t[0], t[1] = t[0], v, i
                elif v > t[2]:
                    t[2] = v
        per_order.append((counters, top))
    lens = np.array([len(d) for d in docs])
    scores = []
    for h in range(k):
        matches: list[int] = []
        totals: list[int] = []
        for counters, top in per_order:
            m = 0
            for g, v in counters[h].items():
                t = top[g]
                ref_max = t[2] if t[1] == h else t[0]
                m += min(v, ref_max)
            matches.append(m)
            totals.append(sum(counters[h].values()))
        others = np.delete(lens, h)
        scores.append(_bleu_from_counts(matches, totals, int(lens[h]), others.tolist()))
    return float(np.mean(scores))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, bit-parallel over ``a``.

    Uses the classic bit-vector recurrence ``V = (V + U) | (V - U)`` with
    ``U = V & match(y)``, holding the rows of the dynamic-programming table as
    Python integers, so each symbol of ``b`` costs a few big-integer
    operations instead of ``len(a)`` cell updates. The tests check it against
    the plain dynamic programme.
    """
    if not a or not b:
        return 0
    masks: dict[str, int] = {}
    for i, x in enumerate(a):
        masks[x] = masks.get(x, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for y in b:
        u = v & masks.get(y, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def rouge_l(a: Sequence[str], b: Sequence[str]) -> float:
    """ROUGE-L F1 between two token sequences (symmetric)."""
    lcs = lcs_length(a, b)
    if lcs == 0:
        return 0.0
    p = lcs / len(a)
    r = lcs / len(b)
    return 2 * p * r / (p + r)


def sample_pairs(k: int, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Every pair when there are at most ``n_pairs``, else ``n_pairs`` random pairs.

    Random pairs are drawn with replacement; each joins two different texts.
    """
    total = k * (k - 1) // 2
    if total <= n_pairs:
        return [(i, j) for i in range(k) for j in range(i + 1, k)]
    i = rng.integers(0, k, size=n_pairs)
    j = rng.integers(0, k - 1, size=n_pairs)
    j = np.where(j >= i, j + 1, j)
    return list(zip(i.tolist(), j.tolist(), strict=True))


def rouge_l_homogenization(
    token_lists: Sequence[Sequence[str]], n_pairs: int, rng: np.random.Generator
) -> float:
    """Mean ROUGE-L F1 over pairs of different non-empty texts (see :func:`sample_pairs`)."""
    docs = [t for t in token_lists if len(t)]
    if len(docs) < 2:
        return math.nan
    pairs = sample_pairs(len(docs), n_pairs, rng)
    return float(np.mean([rouge_l(docs[i], docs[j]) for i, j in pairs]))
