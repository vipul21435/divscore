import math
import random

import numpy as np
import pytest

from divscore.vendi import cosine_kernel, hashed_counts, ngram_vendi, vendi_from_kernel


def test_identical_texts_give_one():
    assert ngram_vendi([["a", "b", "c"]] * 7) == pytest.approx(1.0)


def test_orthogonal_texts_give_n():
    texts = [[f"w{i}a", f"w{i}b", f"w{i}c"] for i in range(9)]
    assert ngram_vendi(texts) == pytest.approx(9.0)


def test_two_groups_of_copies_give_two():
    texts = [["x", "y"]] * 4 + [["p", "q"]] * 4
    assert ngram_vendi(texts) == pytest.approx(2.0)


def test_kernel_from_identity_and_ones():
    assert vendi_from_kernel(np.eye(5)) == pytest.approx(5.0)
    assert vendi_from_kernel(np.ones((5, 5))) == pytest.approx(1.0)
    assert math.isnan(vendi_from_kernel(np.zeros((0, 0))))


def test_sparse_kernel_matches_dense_cosine():
    rng = random.Random(4)
    texts = [[rng.choice("abcdefghijkl") for _ in range(rng.randint(2, 15))] for _ in range(25)]
    vecs = [hashed_counts(t) for t in texts]
    buckets = sorted({b for v in vecs for b in v})
    dense = np.array([[v.get(b, 0) for b in buckets] for v in vecs], dtype=float)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    assert np.allclose(cosine_kernel(vecs), dense @ dense.T)


def test_vendi_between_one_and_n_and_ignores_empty():
    rng = random.Random(9)
    texts = [[rng.choice("abcdef") for _ in range(8)] for _ in range(15)]
    v = ngram_vendi(texts + [[]])
    assert 1.0 <= v <= 15.0
    assert math.isnan(ngram_vendi([[]]))


def test_hashing_is_stable():
    assert hashed_counts(["a", "b"]) == hashed_counts(["a", "b"])
    assert sum(hashed_counts(["a", "b", "c"], ns=(1, 2)).values()) == 5
