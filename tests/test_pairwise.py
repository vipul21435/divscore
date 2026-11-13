import math
import random
from functools import cache

import numpy as np
import pytest
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu as nltk_bleu
from rouge_score.rouge_scorer import _score_lcs

from divscore.pairwise import (
    lcs_length,
    rouge_l,
    rouge_l_homogenization,
    sample_pairs,
    self_bleu,
    sentence_bleu,
)

SMOOTH = SmoothingFunction().method1


def random_texts(seed, k=12, vocab="abcdefg", lo=1, hi=14):
    rng = random.Random(seed)
    return [[rng.choice(vocab) for _ in range(rng.randint(lo, hi))] for _ in range(k)]


@pytest.mark.parametrize("seed", range(8))
def test_sentence_bleu_matches_nltk(seed):
    texts = random_texts(seed)
    hyp, refs = texts[0], texts[1:5]
    for max_n in (2, 4):
        w = tuple([1.0 / max_n] * max_n)
        expected = nltk_bleu(refs, hyp, weights=w, smoothing_function=SMOOTH)
        assert sentence_bleu(hyp, refs, max_n=max_n) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_self_bleu_matches_leave_one_out_nltk(seed):
    texts = random_texts(seed + 100, k=9)
    expected = np.mean(
        [
            nltk_bleu(texts[:i] + texts[i + 1 :], texts[i], smoothing_function=SMOOTH)
            for i in range(len(texts))
        ]
    )
    assert self_bleu(texts) == pytest.approx(expected, abs=1e-12)


def test_self_bleu_closed_forms():
    same = [["the", "cat", "sat", "on", "the", "mat"]] * 5
    assert self_bleu(same) == pytest.approx(1.0)
    disjoint = [[f"w{i}_{j}" for j in range(6)] for i in range(5)]
    assert self_bleu(disjoint) == 0.0
    assert math.isnan(self_bleu([["a"]]))
    with pytest.raises(ValueError):
        sentence_bleu(["a"], [])


def test_self_bleu_falls_as_texts_diverge():
    rng = random.Random(1)
    base = [rng.choice("abcdefghij") for _ in range(20)]
    scores = []
    for flip in (0.0, 0.2, 0.5, 0.9):
        texts = []
        for _ in range(10):
            texts.append([t if rng.random() > flip else f"x{rng.random()}" for t in base])
        scores.append(self_bleu(texts))
    assert scores == sorted(scores, reverse=True)


@cache
def lcs_dp(a, b):
    if not a or not b:
        return 0
    if a[-1] == b[-1]:
        return lcs_dp(a[:-1], b[:-1]) + 1
    return max(lcs_dp(a[:-1], b), lcs_dp(a, b[:-1]))


@pytest.mark.parametrize("seed", range(20))
def test_lcs_bit_parallel_matches_dynamic_programming(seed):
    rng = random.Random(seed)
    a = tuple(rng.choice("abcd") for _ in range(rng.randint(0, 12)))
    b = tuple(rng.choice("abcd") for _ in range(rng.randint(0, 12)))
    assert lcs_length(a, b) == lcs_dp(a, b)


@pytest.mark.parametrize("seed", range(10))
def test_rouge_l_matches_rouge_score(seed):
    rng = random.Random(seed)
    a = [rng.choice("abcdef") for _ in range(rng.randint(1, 30))]
    b = [rng.choice("abcdef") for _ in range(rng.randint(1, 30))]
    assert rouge_l(a, b) == pytest.approx(_score_lcs(a, b).fmeasure)
    assert rouge_l(a, b) == pytest.approx(rouge_l(b, a))


def test_rouge_l_hand_case():
    # LCS of "a b c d" and "a c d e" is "a c d" = 3; P = R = 3/4
    assert rouge_l(list("abcd"), list("acde")) == pytest.approx(0.75)
    assert rouge_l(list("abc"), list("xyz")) == 0.0


def test_sample_pairs_all_or_random():
    rng = np.random.default_rng(0)
    assert sample_pairs(4, 100, rng) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pairs = sample_pairs(50, 200, rng)
    assert len(pairs) == 200
    assert all(i != j and 0 <= i < 50 and 0 <= j < 50 for i, j in pairs)


def test_homogenization_closed_forms_and_seed():
    same = [["a", "b", "c"]] * 6
    rng = np.random.default_rng(0)
    assert rouge_l_homogenization(same, 10, rng) == pytest.approx(1.0)
    disjoint = [[f"t{i}"] * 3 for i in range(6)]
    assert rouge_l_homogenization(disjoint, 10, rng) == 0.0
    texts = random_texts(5, k=40)
    a = rouge_l_homogenization(texts, 50, np.random.default_rng(3))
    b = rouge_l_homogenization(texts, 50, np.random.default_rng(3))
    assert a == b
    assert math.isnan(rouge_l_homogenization([["a"]], 5, rng))
