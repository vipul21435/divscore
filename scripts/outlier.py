"""How much the longest dolly instruction moves the dolly vs alpaca comparison.

    python scripts/outlier.py DATA_DIR

Repeats the token-budget draws of `divscore compare` (20,000 tokens, 100
subsamples, seeds 0 and 1), reports how many dolly draws contain the longest
instruction and what they score, then compares again with that one record left
out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from divscore.budget import subsample_values
from divscore.io import load_texts
from divscore.measure import differences, measure, metric_suite, prepare
from divscore.metrics import distinct_n
from divscore.text import make_docs

BUDGET = 20_000
RESAMPLES = 100


def load(path: Path) -> list:
    return prepare(make_docs(load_texts(path, field="instruction").texts))


def main() -> None:
    data = Path(sys.argv[1])
    dolly = load(data / "dolly.jsonl")
    alpaca = load(data / "alpaca.jsonl")
    lengths = np.array([len(d) for d in dolly])
    longest = dolly[int(lengths.argmax())]
    print(f"dolly instructions: median {np.median(lengths):.0f} tokens, longest {len(longest)}")

    def has_longest(docs: list) -> float:
        return float(any(len(d) > 100 and longest.tokens[: len(d)] == d.tokens for d in docs))

    vals = subsample_values(
        dolly,
        BUDGET,
        {"has": has_longest, "d3": lambda docs: distinct_n([d.tokens for d in docs], 3)},
        RESAMPLES,
        seed=0,
    )
    has = np.array(vals["has"], dtype=bool)
    d3 = np.array(vals["d3"])
    print(f"draws containing it: {has.sum()} of {RESAMPLES}")
    print(f"distinct-3 mean with it {d3[has].mean():.3f}, without it {d3[~has].mean():.3f}")

    suite = metric_suite()
    keep = {k: suite[k] for k in ("distinct-2", "distinct-3")}
    rest = [d for d in dolly if d is not longest]
    a = measure(rest, BUDGET, "dolly without longest", RESAMPLES, 0, metrics=keep)
    b = measure(alpaca, BUDGET, "alpaca", RESAMPLES, 1, metrics=keep)
    for d in differences(a, b):
        print(f"{d.metric}: dolly minus alpaca {d.diff:+.4f} [{d.lo:+.4f}, {d.hi:+.4f}]")


if __name__ == "__main__":
    main()
