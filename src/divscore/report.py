"""Markdown and JSON renderings of a measurement."""

from __future__ import annotations

import json
import math

from .measure import Measurement, higher_is_more_diverse


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def measurement_json(m: Measurement) -> str:
    return json.dumps(m.to_dict(), indent=2, allow_nan=False)


def measurement_markdown(m: Measurement) -> str:
    level = next(iter(m.controlled.values())).level if m.controlled else 0.95
    pct = f"{100 * level:g} %"
    lines = [
        f"## {m.name or 'dataset'}",
        "",
        f"{m.n_texts} texts, {m.n_tokens} tokens ({_fmt(m.mean_tokens, 1)} per text)"
        + (f", {m.skipped} records skipped" if m.skipped else "")
        + ".",
        f"Budget: {m.budget} tokens, {m.resamples} subsamples, seed {m.seed}.",
        "",
        f"| metric | better | whole dataset | at budget | {pct} interval | sd |",
        "|---|---|---:|---:|---|---:|",
    ]
    for name, est in m.controlled.items():
        better = "higher" if higher_is_more_diverse(name) else "lower"
        lines.append(
            f"| {name} | {better} | {_fmt(m.naive[name])} | {_fmt(est.mean)} | "
            f"[{_fmt(est.lo)}, {_fmt(est.hi)}] | {_fmt(est.sd)} |"
        )
    lines.append("")
    return "\n".join(lines)
