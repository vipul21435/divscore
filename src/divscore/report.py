"""Markdown and JSON renderings of measurements, comparisons and openings."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from .measure import Difference, Measurement, higher_is_more_diverse


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def _better(name: str) -> str:
    return "higher" if higher_is_more_diverse(name) else "lower"


def to_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=False)


def measurement_json(m: Measurement) -> str:
    return to_json(m.to_dict())


def _budget_label(m: Measurement, unit: str) -> str:
    return f"{m.token_budget} tokens" if unit == "tokens" else f"{m.text_budget} texts"


def measurement_markdown(m: Measurement) -> str:
    level = next(iter(m.controlled.values())).level if m.controlled else 0.95
    pct = f"{100 * level:g} %"
    lens = m.lengths
    lines = [
        f"## {m.name or 'dataset'}",
        "",
        f"{m.n_texts} texts, {m.n_tokens} tokens"
        + (f", {m.skipped} records skipped" if m.skipped else "")
        + (f", every text cut to {m.truncate} tokens" if m.truncate else "")
        + ".",
    ]
    if lens:
        lines.append(
            f"Tokens per text: mean {_fmt(lens['mean'], 1)}, median {_fmt(lens['median'], 0)}, "
            f"10th-90th percentile {_fmt(lens['p10'], 0)}-{_fmt(lens['p90'], 0)}, "
            f"max {_fmt(lens['max'], 0)}."
        )
    lines.append(f"{m.resamples} subsamples per budget, seed {m.seed}.")
    for unit in ("tokens", "texts"):
        names = [k for k, u in m.units.items() if u == unit]
        if not names:
            continue
        lines += [
            "",
            f"Budget: {_budget_label(m, unit)}",
            "",
            f"| metric | better | whole dataset | at budget | {pct} interval | sd |",
            "|---|---|---:|---:|---|---:|",
        ]
        for name in names:
            est = m.controlled[name]
            lines.append(
                f"| {name} | {_better(name)} | {_fmt(m.naive.get(name))} | {_fmt(est.mean)} | "
                f"[{_fmt(est.lo)}, {_fmt(est.hi)}] | {_fmt(est.sd)} |"
            )
    lines.append("")
    return "\n".join(lines)


def comparison_json(a: Measurement, b: Measurement, diffs: Sequence[Difference]) -> str:
    return to_json(
        {"a": a.to_dict(), "b": b.to_dict(), "differences": [d.to_dict() for d in diffs]}
    )


def comparison_markdown(a: Measurement, b: Measurement, diffs: Sequence[Difference]) -> str:
    level = next(iter(a.controlled.values())).level if a.controlled else 0.95
    pct = f"{100 * level:g} %"
    lines = [
        f"## {a.name} (A) vs {b.name} (B)",
        "",
        f"A: {a.n_texts} texts, {a.n_tokens} tokens. B: {b.n_texts} texts, {b.n_tokens} tokens.",
        f"{a.resamples} subsamples per side and budget, seed {a.seed}."
        + (f" Every text cut to {a.truncate} tokens." if a.truncate else ""),
    ]
    for unit in ("tokens", "texts"):
        rows = [d for d in diffs if d.unit == unit]
        if not rows:
            continue
        lines += [
            "",
            f"Budget: {_budget_label(a, unit)}",
            "",
            f"| metric | better | A | B | A - B | {pct} interval | whole data A - B "
            "| more diverse |",
            "|---|---|---:|---:|---:|---|---:|---|",
        ]
        for d in rows:
            naive = (
                _fmt(d.naive_a - d.naive_b)
                if d.naive_a is not None and d.naive_b is not None
                else "n/a"
            )
            if not d.clear:
                verdict = "unclear"
            else:
                a_more = (d.diff > 0) == higher_is_more_diverse(d.metric)
                verdict = "A" if a_more else "B"
            lines.append(
                f"| {d.metric} | {_better(d.metric)} | {_fmt(d.a)} | {_fmt(d.b)} | "
                f"{_fmt(d.diff)} | [{_fmt(d.lo)}, {_fmt(d.hi)}] | {naive} | {verdict} |"
            )
    lines.append("")
    return "\n".join(lines)


def openings_markdown(name: str, summary: dict[str, Any]) -> str:
    lines = [
        f"## {name}: openings",
        "",
        f"{summary['n_texts']} texts; first {summary['words']} words. "
        f"Top-{summary['top']} share {_fmt(summary['top_share'])}, "
        f"Herfindahl index {_fmt(summary['herfindahl'])}, "
        f"{summary['distinct']} distinct openings.",
        "",
        "| rank | opening | texts | share |",
        "|---:|---|---:|---:|",
    ]
    for i, row in enumerate(summary["openings"], 1):
        lines.append(f"| {i} | {row['text']} | {row['count']} | {_fmt(row['share'])} |")
    if summary.get("templates"):
        lines += [
            "",
            f"Most frequent {summary['template_n']}-token templates "
            "(numbers and quoted spans masked; texts containing each):",
            "",
            "| rank | template | texts | share |",
            "|---:|---|---:|---:|",
        ]
        for i, row in enumerate(summary["templates"], 1):
            lines.append(f"| {i} | {row['text']} | {row['count']} | {_fmt(row['share'])} |")
    lines.append("")
    return "\n".join(lines)
