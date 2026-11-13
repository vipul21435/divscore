import math
from collections import Counter

import numpy as np
import pytest

from divscore.structure import (
    geometric_points,
    heaps_beta,
    heaps_fit,
    herfindahl,
    length_summary,
    mask_tokens,
    opening,
    opening_counts,
    ranked,
    template_counts,
    template_share,
    top_share,
    type_token_curve,
)
from divscore.text import tokenize


def test_opening_skips_punctuation():
    assert opening(tokenize('"Hello, there my friend"')) == ("hello", "there", "my")
    assert opening(tokenize("Hi!")) == ("hi",)
    assert opening(tokenize("?!")) is None


def test_top_share_and_herfindahl_closed_forms():
    texts = [tokenize("Write a poem about x")] * 6 + [tokenize("Explain why y")] * 2
    counts = opening_counts(texts)
    assert counts[("write", "a", "poem")] == 6
    assert top_share(counts, top=1) == pytest.approx(0.75)
    assert herfindahl(counts) == pytest.approx(0.75**2 + 0.25**2)
    unique = opening_counts([[f"a{i}", "b", "c"] for i in range(10)])
    assert herfindahl(unique) == pytest.approx(0.1)
    assert math.isnan(top_share(Counter()))
    assert math.isnan(herfindahl(Counter()))


def test_mask_numbers_and_quotes():
    toks = mask_tokens('Sort 3, 1.5 and 20 in "this list" for “me” now')
    assert toks == ["sort", "<num>", ",", "<num>", "and", "<num>", "in", "<q>", "for", "<q>", "now"]
    # apostrophes inside words are not quotes
    assert mask_tokens("don't stop it's 'here'") == ["don't", "stop", "it's", "<q>"]


def test_templates_after_masking():
    texts = [
        mask_tokens(f'Convert {n} miles to km for "{w}" please') for n, w in [(3, "a"), (7, "b")]
    ]
    counts = template_counts(texts, n=5)
    assert counts[("convert", "<num>", "miles", "to", "km")] == 2
    assert template_share(texts, n=5, top=1) == 1.0
    assert math.isnan(template_share([[]]))


def test_type_token_curve_and_heaps_closed_form():
    # every token new: V = N exactly, so beta = 1 and K = 1
    toks = [[f"t{i}" for i in range(1000)]]
    ns, vs = type_token_curve(toks, [10, 100, 1000])
    assert ns.tolist() == [10, 100, 1000]
    assert vs.tolist() == [10, 100, 1000]
    k, beta = heaps_fit(ns, vs)
    assert beta == pytest.approx(1.0)
    assert k == pytest.approx(1.0)
    assert heaps_beta(toks) == pytest.approx(1.0)


def test_heaps_recovers_exponent_of_power_law():
    ns = np.array([100, 1000, 10000, 100000], dtype=float)
    vs = 7.0 * ns**0.6
    k, beta = heaps_fit(ns, vs)
    assert beta == pytest.approx(0.6)
    assert k == pytest.approx(7.0)
    assert math.isnan(heaps_fit(ns[:1], vs[:1])[1])
    assert math.isnan(heaps_beta([["a"] * 10]))


def test_heaps_beta_lower_for_repetitive_text():
    rng = np.random.default_rng(0)
    rich = [[f"w{x}" for x in rng.integers(0, 50000, 4000)]]
    poor = [[f"w{x}" for x in rng.integers(0, 300, 4000)]]
    assert heaps_beta(rich) > heaps_beta(poor)


def test_geometric_points_and_lengths():
    pts = geometric_points(10, 1000, 3)
    assert pts == [10, 100, 1000]
    assert geometric_points(10, 5) == [5]
    s = length_summary([1, 2, 3, 4, 100])
    assert s["median"] == 3.0
    assert s["max"] == 100.0
    assert length_summary([]) == {}


def test_template_share_ignores_document_order_on_ties():
    # x is in three texts; p and q tie at two. Ranking ties by insertion order
    # made the answer depend on the order of the texts (and on hash order).
    docs = [["x", "p"], ["x", "p"], ["x"], ["q"], ["q"]]
    forward = template_share(docs, n=1, top=2)
    backward = template_share(docs[::-1], n=1, top=2)
    assert forward == backward == pytest.approx(3 / 5)


def test_ranked_breaks_ties_by_key():
    a = Counter({("b",): 2, ("a",): 2, ("c",): 5})
    b = Counter({("a",): 2, ("c",): 5, ("b",): 2})
    assert ranked(a, 2) == ranked(b, 2) == [(("c",), 5), (("a",), 2)]
    assert ranked(a) == [(("c",), 5), (("a",), 2), (("b",), 2)]


def test_mask_tokens_keeps_case_when_asked():
    assert mask_tokens("Call Me 3 times", lowercase=False) == ["Call", "Me", "<num>", "times"]
    assert mask_tokens("İstanbul 3") == [tokenize("İstanbul")[0], "<num>"]


def test_digits_inside_words_are_not_masked():
    text = "utf8 h2o x86_64 python3.12 3rd"
    assert mask_tokens(text) == tokenize(text)
    assert mask_tokens("costs 3.50 or (12) items, 4:30") == [
        "costs",
        "<num>",
        "or",
        "(",
        "<num>",
        ")",
        "items",
        ",",
        "<num>",
    ]


def test_identifier_corpus_is_not_one_template():
    texts = [mask_tokens(f"open file w{i} then close w{i + 1} now") for i in range(20)]
    assert template_share(texts, n=5, top=1) < 0.2
