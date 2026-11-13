import pytest

from divscore.measure import higher_is_more_diverse, measure, metric_suite
from divscore.report import measurement_json, measurement_markdown
from divscore.text import make_docs


def test_suite_names():
    names = list(metric_suite())
    assert names == [
        "distinct-1",
        "distinct-2",
        "distinct-3",
        "entropy-1",
        "entropy-2",
        "cr-gzip",
        "cr-bz2",
    ]
    assert list(metric_suite(ns=(2,), compressors=("lzma",))) == [
        "distinct-2",
        "entropy-1",
        "entropy-2",
        "cr-lzma",
    ]


def test_direction():
    assert higher_is_more_diverse("distinct-4")
    assert higher_is_more_diverse("entropy-1")
    assert not higher_is_more_diverse("cr-gzip")
    with pytest.raises(KeyError):
        higher_is_more_diverse("unknown-1")


def test_measure_counts_and_empty_texts():
    docs = make_docs(["a b c", "", "d e f g"])
    m = measure(docs, budget=5, resamples=5, skipped=1)
    assert m.n_texts == 2
    assert m.n_tokens == 7
    assert m.skipped == 2
    assert m.mean_tokens == pytest.approx(3.5)
    assert set(m.naive) == set(m.controlled)


def test_reports_render():
    docs = make_docs(["one two three", "four five six seven"])
    m = measure(docs, budget=6, name="tiny", resamples=3)
    md = measurement_markdown(m)
    assert md.startswith("## tiny")
    assert "95 % interval" in md
    js = measurement_json(m)
    assert '"name": "tiny"' in js
