import math

import pytest

from divscore.measure import (
    differences,
    higher_is_more_diverse,
    measure,
    metric_suite,
    prepare,
    text_suite,
)
from divscore.report import (
    comparison_json,
    comparison_markdown,
    measurement_json,
    measurement_markdown,
)
from divscore.text import make_docs


def test_suite_names():
    assert list(metric_suite()) == [
        "distinct-1",
        "distinct-2",
        "distinct-3",
        "entropy-1",
        "entropy-2",
        "cr-gzip",
        "cr-bz2",
        "heaps-beta",
    ]
    assert list(metric_suite(ns=(2,), compressors=("lzma",)))[:2] == ["distinct-2", "entropy-1"]
    assert list(text_suite()) == [
        "self-bleu",
        "rouge-l",
        "vendi",
        "near-dup",
        "opening-top10",
        "opening-hhi",
        "template-top10",
    ]


def test_direction():
    assert higher_is_more_diverse("distinct-4")
    assert higher_is_more_diverse("entropy-1")
    assert higher_is_more_diverse("vendi")
    assert not higher_is_more_diverse("cr-gzip")
    assert not higher_is_more_diverse("self-bleu")
    with pytest.raises(KeyError):
        higher_is_more_diverse("unknown-1")


def test_prepare_drops_empty_and_truncates():
    docs = make_docs(["a b c d", "", "e"])
    assert [len(d) for d in prepare(docs, truncate=2)] == [2, 1]
    with pytest.raises(ValueError):
        prepare(docs, truncate=0)


def test_measure_counts_and_empty_texts():
    docs = make_docs(["a b c", "", "d e f g"])
    m = measure(docs, budget=5, resamples=5, skipped=1)
    assert m.n_texts == 2
    assert m.n_tokens == 7
    assert m.skipped == 2
    assert m.mean_tokens == pytest.approx(3.5)
    assert set(m.naive) == set(m.controlled)
    assert m.text_budget is None
    with pytest.raises(ValueError):
        measure(docs, budget=5, level=2.0)


def test_text_budget_metrics_on_identical_texts():
    docs = make_docs(["write a short poem about the sea"] * 12)
    m = measure(docs, budget=20, text_budget=6, resamples=4)
    assert m.controlled["vendi"].mean == pytest.approx(1.0)
    assert m.controlled["self-bleu"].mean == pytest.approx(1.0)
    assert m.controlled["rouge-l"].mean == pytest.approx(1.0)
    assert m.controlled["near-dup"].mean == pytest.approx(1.0)
    assert m.controlled["opening-hhi"].mean == pytest.approx(1.0)
    assert m.naive["opening-top10"] == pytest.approx(1.0)
    assert m.naive["vendi"] is None
    assert m.units["vendi"] == "texts"


def test_differences_and_reports():
    a_docs = make_docs([f"alpha{i} beta{i} gamma{i}" for i in range(20)])
    a = measure(a_docs, 30, "a", 10, 0, text_budget=8)
    b = measure(make_docs(["same words here"] * 20), 30, "b", 10, 1, text_budget=8)
    diffs = {d.metric: d for d in differences(a, b)}
    assert diffs["distinct-1"].diff > 0
    assert diffs["distinct-1"].clear
    assert diffs["vendi"].diff == pytest.approx(7.0)
    assert diffs["vendi"].naive_a is None
    md = comparison_markdown(a, b, list(diffs.values()))
    assert "## a (A) vs b (B)" in md
    assert "| A |" in md
    js = comparison_json(a, b, list(diffs.values()))
    assert '"interval_excludes_zero": true' in js


def test_reports_render():
    docs = make_docs(["one two three", "four five six seven", "eight nine"])
    m = measure(docs, budget=6, name="tiny", resamples=3, text_budget=2, truncate=3)
    md = measurement_markdown(m)
    assert md.startswith("## tiny")
    assert "95 % interval" in md
    assert "every text cut to 3 tokens" in md
    js = measurement_json(m)
    assert '"name": "tiny"' in js
    assert not math.isnan(m.lengths["mean"])


def test_whole_dataset_heaps_beta_ignores_file_order():
    import random

    from divscore.structure import heaps_beta_shuffled

    rng = random.Random(0)
    # a file sorted by "topic": the first half uses one vocabulary, the second another
    texts = [" ".join(f"a{rng.randrange(300)}" for _ in range(12)) for _ in range(60)]
    texts += [" ".join(f"b{rng.randrange(3000)}" for _ in range(12)) for _ in range(60)]
    docs = make_docs(texts)
    fwd = measure(docs, budget=200, resamples=2, seed=3)
    rev = measure(docs[::-1], budget=200, resamples=2, seed=3)
    assert fwd.naive["heaps-beta"] == rev.naive["heaps-beta"]
    expected = heaps_beta_shuffled([d.tokens for d in docs], seed=3)
    assert fwd.naive["heaps-beta"] == pytest.approx(expected)


def test_differences_refuses_equal_seeds():
    docs = make_docs([f"t{i} u{i}" for i in range(10)])
    a = measure(docs, 6, "a", 3, seed=5)
    b = measure(docs, 6, "b", 3, seed=5)
    with pytest.raises(ValueError, match="same seed"):
        differences(a, b)
