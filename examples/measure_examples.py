"""Measure the two small example files at the same token budget.

Run from the repository root:

    python examples/measure_examples.py
"""

from pathlib import Path

from divscore.io import load_texts
from divscore.measure import measure
from divscore.report import measurement_markdown
from divscore.text import make_docs

DATA = Path(__file__).resolve().parent / "data"
BUDGET = 250

for name in ("varied.jsonl", "templated.jsonl"):
    docs = make_docs(load_texts(DATA / name).texts)
    print(measurement_markdown(measure(docs, budget=BUDGET, name=name, resamples=200, seed=0)))
