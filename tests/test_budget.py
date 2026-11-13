import math

import numpy as np
import pytest

from divscore.budget import at_budget, sample_to_budget, summarize
from divscore.metrics import distinct_n
from divscore.text import make_docs, total_tokens


def zipf_corpus(n_texts, seed, vocab=2000, length=(5, 30)):
    """Texts drawn from one Zipf-like word distribution."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, vocab + 1)
    p = 1.0 / ranks
    p /= p.sum()
    texts = []
    for _ in range(n_texts):
        k = int(rng.integers(length[0], length[1] + 1))
        texts.append(" ".join(f"w{i}" for i in rng.choice(vocab, size=k, p=p)))
    return make_docs(texts)


def d2(docs):
    return distinct_n([d.tokens for d in docs], 2)


def test_sample_hits_budget_exactly():
    docs = make_docs(["a b c", "", "d e", "f g h i"])
    rng = np.random.default_rng(1)
    for budget in (1, 4, 5, 9):
        sample = sample_to_budget(docs, budget, rng)
        assert total_tokens(sample) == budget
        assert all(len(d) > 0 for d in sample)


def test_sample_truncated_text_matches_tokens():
    docs = make_docs(["Alpha beta gamma delta epsilon."])
    sample = sample_to_budget(docs, 3, np.random.default_rng(0))
    assert sample[0].text == "Alpha beta gamma"


def test_budget_too_large_or_zero():
    docs = make_docs(["a b"])
    with pytest.raises(ValueError, match="exceeds"):
        sample_to_budget(docs, 3, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_to_budget(docs, 0, np.random.default_rng(0))


def test_at_budget_deterministic_under_seed():
    docs = zipf_corpus(300, seed=5)
    a = at_budget(docs, 1000, {"d2": d2}, resamples=20, seed=7)
    b = at_budget(docs, 1000, {"d2": d2}, resamples=20, seed=7)
    c = at_budget(docs, 1000, {"d2": d2}, resamples=20, seed=8)
    assert a == b
    assert a != c


def test_full_budget_has_no_spread_for_order_free_metric():
    docs = make_docs(["a b c", "a b d", "e f"])
    est = at_budget(docs, total_tokens(docs), {"d2": d2}, resamples=10)["d2"]
    assert est.sd == pytest.approx(0.0)
    assert est.lo == pytest.approx(est.hi)
    assert est.mean == pytest.approx(d2(docs))


def test_budget_removes_the_size_effect():
    # Same distribution, ten times more text: the naive number drops a lot,
    # the number at a common budget does not.
    small = zipf_corpus(200, seed=1)
    large = zipf_corpus(2000, seed=2)
    assert d2(large) < d2(small) - 0.1
    budget = 2000
    es = at_budget(small, budget, {"d2": d2}, resamples=40, seed=0)["d2"]
    el = at_budget(large, budget, {"d2": d2}, resamples=40, seed=0)["d2"]
    assert abs(es.mean - el.mean) < 0.02
    assert es.lo <= el.mean <= es.hi or el.lo <= es.mean <= el.hi


def test_distinct_falls_with_budget_on_average():
    docs = zipf_corpus(800, seed=4)
    budgets = (500, 2000, 8000)
    means = [at_budget(docs, b, {"d2": d2}, resamples=15, seed=0)["d2"].mean for b in budgets]
    assert means[0] > means[1] > means[2]


def test_summarize_ignores_nan_and_handles_all_nan():
    est = summarize([1.0, math.nan, 3.0], level=0.5)
    assert est.mean == pytest.approx(2.0)
    assert est.resamples == 3
    empty = summarize([math.nan, math.nan])
    assert math.isnan(empty.mean)
    assert empty.to_dict()["mean"] is None


def test_at_budget_rejects_bad_arguments():
    docs = make_docs(["a b c"])
    with pytest.raises(ValueError):
        at_budget(docs, 2, {"d2": d2}, resamples=0)
    with pytest.raises(ValueError):
        at_budget(docs, 2, {"d2": d2}, level=1.5)
