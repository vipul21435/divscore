"""Print a README table from the JSON reports in results/.

    python scripts/table.py results instruction
    python scripts/table.py results instruction15 short

``short`` prints only the length rows, the token-budget rows at the budget and
the self-BLEU and Vendi rows.

One column per dataset, one row per metric; token-budget metrics get a row
for the whole dataset and a row at the budget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SETS = ("dolly", "alpaca", "magpie", "personas")
TOKEN_ROWS = ("distinct-1", "distinct-2", "distinct-3", "entropy-2", "cr-gzip", "heaps-beta")
TEXT_ROWS = (
    "self-bleu",
    "rouge-l",
    "vendi",
    "near-dup",
    "opening-top10",
    "opening-hhi",
    "template-top10",
)


def cell(m: dict, digits: int = 3) -> str:
    e = m["at_budget"]
    return f"{e['mean']:.{digits}f} [{e['lo']:.{digits}f}, {e['hi']:.{digits}f}]"


def row(label: str, cells: list[str]) -> str:
    return f"| {label} | " + " | ".join(cells) + " |"


def main() -> None:
    root, kind = Path(sys.argv[1]), sys.argv[2]
    short = len(sys.argv) > 3 and sys.argv[3] == "short"
    reports = [json.loads((root / f"{kind}-{s}.json").read_text()) for s in SETS]
    first = reports[0]
    tb, xb = first["token_budget"], first["text_budget"]
    print(row("metric", list(SETS)))
    print("|---|" + "---|" * len(SETS))
    print(row("texts", [str(r["n_texts"]) for r in reports]))
    print(row("tokens", [str(r["n_tokens"]) for r in reports]))
    print(row("mean tokens per text", [f"{r['lengths']['mean']:.1f}" for r in reports]))
    for key, label in (("median", "median"), ("p90", "90th percentile"), ("max", "longest")):
        print(row(f"{label} tokens per text", [f"{r['lengths'][key]:.0f}" for r in reports]))
    for name in TOKEN_ROWS:
        if not short:
            print(row(f"{name}, whole", [f"{r['metrics'][name]['naive']:.3f}" for r in reports]))
        print(row(f"{name}, {tb} tokens", [cell(r["metrics"][name]) for r in reports]))
    if short:
        for name in ("self-bleu", "vendi"):
            digits = 1 if name == "vendi" else 3
            print(row(f"{name}, {xb} texts", [cell(r["metrics"][name], digits) for r in reports]))
        return
    for name in TEXT_ROWS:
        digits = 1 if name == "vendi" else 3
        print(row(f"{name}, {xb} texts", [cell(r["metrics"][name], digits) for r in reports]))
    for name in ("opening-top10", "opening-hhi"):
        print(row(f"{name}, whole", [f"{r['metrics'][name]['naive']:.3f}" for r in reports]))


if __name__ == "__main__":
    main()
