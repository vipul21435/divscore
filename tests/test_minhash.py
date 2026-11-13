import math
import random

import numpy as np
import pytest

from divscore.minhash import (
    MinHasher,
    jaccard,
    near_duplicate_pairs,
    near_duplicate_rate,
    shingle_hashes,
)


def words(rng, n, vocab=400):
    return [f"w{rng.randrange(vocab)}" for _ in range(n)]


def test_shingles():
    assert shingle_hashes(["a", "b", "c"], k=5).size == 1
    assert shingle_hashes(list("abcdefg"), k=5).size == 3
    assert shingle_hashes(["a", "a", "a", "a", "a", "a"], k=5).size == 1
    assert shingle_hashes([], k=5).size == 0


def test_signature_estimates_jaccard():
    rng = random.Random(0)
    hasher = MinHasher(num_perm=256, seed=1)
    for _ in range(10):
        a = set(rng.sample(range(1000), 120))
        b = set(rng.sample(sorted(a), 60)) | set(rng.sample(range(1000, 2000), 60))
        sa = hasher.signature(np.array(sorted(a), dtype=np.uint64))
        sb = hasher.signature(np.array(sorted(b), dtype=np.uint64))
        assert float(np.mean(sa == sb)) == pytest.approx(jaccard(a, b), abs=0.1)


def test_jaccard_edge_cases():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, {2}) == 0.0


def test_rate_closed_forms():
    rng = random.Random(2)
    distinct = [words(rng, 30) for _ in range(20)]
    assert near_duplicate_rate(distinct) == 0.0
    assert near_duplicate_rate([distinct[0]] * 5) == 1.0
    # one exact copy among ten distinct texts flags two of eleven
    texts = distinct[:10] + [distinct[3]]
    assert near_duplicate_rate(texts) == pytest.approx(2 / 11)
    assert math.isnan(near_duplicate_rate([[]]))


def test_one_word_edit_is_found_and_unrelated_is_not():
    rng = random.Random(3)
    base = words(rng, 200)
    edited = list(base)
    edited[100] = "changed"
    other = words(rng, 200)
    pairs = near_duplicate_pairs([base, edited, other], threshold=0.8)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_bands_must_divide():
    with pytest.raises(ValueError):
        near_duplicate_pairs([["a"]], num_perm=100, bands=32)
