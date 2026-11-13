"""Measure and compare the two small example files at the same budgets.

Run from the repository root:

    python examples/measure_examples.py
"""

from pathlib import Path

from divscore.io import load_texts
from divscore.measure import differences, measure
from divscore.report import comparison_markdown, measurement_markdown
from divscore.text import make_docs

DATA = Path(__file__).resolve().parent / "data"
TOKEN_BUDGET = 140
TEXT_BUDGET = 15

results = []
for seed, name in enumerate(("varied.jsonl", "templated.jsonl")):
    docs = make_docs(load_texts(DATA / name).texts)
    m = measure(
        docs, budget=TOKEN_BUDGET, text_budget=TEXT_BUDGET, name=name, resamples=200, seed=seed
    )
    print(measurement_markdown(m))
    results.append(m)

print(comparison_markdown(results[0], results[1], differences(results[0], results[1])))
