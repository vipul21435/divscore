import math
import random

import pytest

from divscore.metrics import (
    compression_ratio,
    distinct_n,
    ngram_counts,
    ngram_entropy,
    ngrams,
)


def test_ngrams_basic():
    assert list(ngrams(["a", "b", "c"], 2)) == [("a", "b"), ("b", "c")]
    assert list(ngrams(["a"], 2)) == []
    with pytest.raises(ValueError):
        list(ngrams(["a"], 0))


def test_distinct_n_by_hand():
    texts = [["the", "cat", "sat"], ["the", "cat", "ran"]]
    # unigrams: the, cat, sat, the, cat, ran -> 4 distinct of 6
    assert distinct_n(texts, 1) == pytest.approx(4 / 6)
    # bigrams: (the cat) (cat sat) (the cat) (cat ran) -> 3 distinct of 4
    assert distinct_n(texts, 2) == pytest.approx(3 / 4)
    # trigrams: (the cat sat) (the cat ran) -> 2 of 2
    assert distinct_n(texts, 3) == pytest.approx(1.0)


def test_ngrams_do_not_cross_texts():
    # joined, "b c" would be a bigram; split, it is not
    counts = ngram_counts([["a", "b"], ["c", "d"]], 2)
    assert ("b", "c") not in counts
    assert sum(counts.values()) == 2


def test_distinct_n_identical_texts_and_empty():
    same = [["x", "y", "z"]] * 10
    assert distinct_n(same, 1) == pytest.approx(3 / 30)
    assert math.isnan(distinct_n([["a"]], 2))
    assert math.isnan(distinct_n([], 1))


def test_entropy_closed_forms():
    # uniform over k types gives log2 k bits
    assert ngram_entropy([["a", "b", "c", "d"]], 1) == pytest.approx(2.0)
    assert ngram_entropy([["a"] * 7], 1) == pytest.approx(0.0)
    # p = (1/2, 1/4, 1/4) gives 1.5 bits
    assert ngram_entropy([["a", "a", "b", "c"]], 1) == pytest.approx(1.5)
    assert math.isnan(ngram_entropy([], 1))


def test_entropy_bounded_by_log_of_distinct():
    rng = random.Random(3)
    texts = [[rng.choice("abcdefgh") for _ in range(20)] for _ in range(30)]
    for n in (1, 2, 3):
        counts = ngram_counts(texts, n)
        assert ngram_entropy(texts, n) <= math.log2(len(counts)) + 1e-12


@pytest.mark.parametrize("method", ["gzip", "bz2", "lzma"])
def test_compression_ratio_orders_repetition(method):
    rng = random.Random(0)
    words = [f"w{i}" for i in range(500)]
    varied = [" ".join(rng.choice(words) for _ in range(12)) for _ in range(200)]
    repeated = ["please write a short poem about the sea"] * 200
    assert compression_ratio(repeated, method) > compression_ratio(varied, method) > 1.0


def test_compression_ratio_deterministic_and_errors():
    texts = ["a b c", "d e f"]
    assert compression_ratio(texts) == compression_ratio(texts)
    assert math.isnan(compression_ratio([]))
    with pytest.raises(ValueError, match="unknown compressor"):
        compression_ratio(texts, "zip")
