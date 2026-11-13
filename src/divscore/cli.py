"""Command-line interface: ``divscore measure | compare | openings | curve``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .io import FORMATS, load_texts
from .measure import differences, measure, prepare
from .report import (
    comparison_json,
    comparison_markdown,
    measurement_json,
    measurement_markdown,
    openings_markdown,
    to_json,
)
from .structure import openings_summary
from .text import Doc, make_docs, total_tokens

DEFAULT_BUDGET = 10_000
DEFAULT_TEXTS = 500


class UsageError(Exception):
    """A problem with the inputs that the user can fix with a flag."""


def _add_input_args(p: argparse.ArgumentParser, second: bool = False) -> None:
    p.add_argument(
        "--field", default="text", help="text field; dotted path, join with + (default: text)"
    )
    p.add_argument("--role", default=None, help="for chat records: take the turns of this role")
    if second:
        p.add_argument("--field-b", default=None, help="text field of B (default: --field)")
        p.add_argument("--role-b", default=None, help="role for B (default: --role)")
    p.add_argument("--input-format", choices=FORMATS, default=None, help="default: from extension")
    p.add_argument("--limit", type=int, default=None, help="read at most this many texts")
    p.add_argument("--keep-case", action="store_true", help="do not lowercase tokens")
    p.add_argument("--truncate", type=int, default=None, help="cut every text to this many tokens")


def _add_budget_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"tokens per subsample for the token metrics (default: {DEFAULT_BUDGET})",
    )
    p.add_argument(
        "--texts",
        type=int,
        default=DEFAULT_TEXTS,
        help=f"texts per subsample for the text metrics; 0 skips them (default: {DEFAULT_TEXTS})",
    )
    p.add_argument(
        "--resamples", type=int, default=100, help="subsamples per budget (default: 100)"
    )
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--level", type=float, default=0.95, help="interval level (default: 0.95)")


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=("md", "json"), default="md", help="report format")
    p.add_argument("-o", "--out", type=Path, default=None, help="write the report here")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divscore",
        description="Diversity of a text dataset, with every metric taken at a fixed budget.",
    )
    parser.add_argument("--version", action="version", version=f"divscore {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="measure one dataset at a token and a text budget")
    m.add_argument("path", type=Path)
    m.add_argument("--name", default=None, help="label in the report (default: file name)")
    _add_input_args(m)
    _add_budget_args(m)
    _add_output_args(m)

    c = sub.add_parser("compare", help="compare two datasets at the same budgets")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--name-a", default=None)
    c.add_argument("--name-b", default=None)
    _add_input_args(c, second=True)
    _add_budget_args(c)
    _add_output_args(c)

    o = sub.add_parser("openings", help="most common openings and masked templates")
    o.add_argument("path", type=Path)
    o.add_argument("--name", default=None)
    o.add_argument("--words", type=int, default=3, help="words in an opening (default: 3)")
    o.add_argument("--top", type=int, default=20, help="openings to list (default: 20)")
    o.add_argument("--template-n", type=int, default=5, help="template length (default: 5)")
    o.add_argument("--templates", type=int, default=10, help="templates to list; 0 skips them")
    _add_input_args(o)
    _add_output_args(o)

    v = sub.add_parser("curve", help="type-token curves with Heaps' law fits (needs matplotlib)")
    v.add_argument("paths", type=Path, nargs="+")
    v.add_argument("--names", default=None, help="comma-separated labels")
    v.add_argument(
        "--max-tokens", type=int, default=None, help="curve length (default: smallest dataset)"
    )
    v.add_argument("--orders", type=int, default=5, help="random text orders to average")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--plot", type=Path, default=None, help="write a PNG plot here")
    v.add_argument("--title", default="")
    _add_input_args(v)
    _add_output_args(v)
    return parser


def _emit(text: str, out: Path | None) -> None:
    text = text if text.endswith("\n") else text + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        out.write_text(text, encoding="utf-8")


def _load(path: Path, field: str, role: str | None, args: argparse.Namespace) -> tuple:
    loaded = load_texts(path, field=field, role=role, fmt=args.input_format, limit=args.limit)
    if not loaded.texts:
        raise UsageError(f"no texts found in {path} (field {field!r})")
    docs = prepare(make_docs(loaded.texts, lowercase=not args.keep_case), args.truncate)
    return loaded, docs


def _check_budgets(path: Path, docs: list[Doc], args: argparse.Namespace) -> None:
    available = total_tokens(docs)
    if available < args.budget:
        raise UsageError(
            f"{path} holds {available} tokens, fewer than the budget of {args.budget}; "
            f"pass --budget {available} or less"
        )
    if args.texts and len(docs) < args.texts:
        raise UsageError(
            f"{path} holds {len(docs)} non-empty texts, fewer than the text budget of "
            f"{args.texts}; pass --texts {len(docs)} or less"
        )
    if 2 * args.budget > available or (args.texts and 2 * args.texts > len(docs)):
        print(
            f"divscore: warning: a budget above half of {path} ({available} tokens, "
            f"{len(docs)} texts) leaves little room between subsamples; intervals narrow "
            "towards zero as the budget nears the dataset size",
            file=sys.stderr,
        )


def _measure_one(path: Path, name: str | None, docs: list[Doc], skipped: int, seed: int, args):
    return measure(
        docs,
        budget=args.budget,
        name=name or path.name,
        resamples=args.resamples,
        seed=seed,
        level=args.level,
        text_budget=args.texts or None,
        skipped=skipped,
        truncate=args.truncate,
        lowercase=not args.keep_case,
    )


def _cmd_measure(args: argparse.Namespace) -> int:
    loaded, docs = _load(args.path, args.field, args.role, args)
    _check_budgets(args.path, docs, args)
    result = _measure_one(args.path, args.name, docs, loaded.skipped, args.seed, args)
    _emit(
        measurement_json(result) if args.format == "json" else measurement_markdown(result),
        args.out,
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    la, da = _load(args.a, args.field, args.role, args)
    field_b = args.field_b or args.field
    role_b = args.role_b if args.role_b is not None else args.role
    lb, db = _load(args.b, field_b, role_b, args)
    _check_budgets(args.a, da, args)
    _check_budgets(args.b, db, args)
    a = _measure_one(args.a, args.name_a, da, la.skipped, args.seed, args)
    b = _measure_one(args.b, args.name_b, db, lb.skipped, args.seed + 1, args)
    diffs = differences(a, b, args.level)
    text = (
        comparison_json(a, b, diffs) if args.format == "json" else comparison_markdown(a, b, diffs)
    )
    _emit(text, args.out)
    return 0


def _cmd_openings(args: argparse.Namespace) -> int:
    loaded, docs = _load(args.path, args.field, args.role, args)
    summary = openings_summary(
        [d.text for d in docs],
        [d.tokens for d in docs],
        words=args.words,
        top=args.top,
        template_n=args.template_n,
        templates=args.templates,
        lowercase=not args.keep_case,
    )
    name = args.name or args.path.name
    if args.format == "json":
        _emit(to_json({"name": name, **summary}), args.out)
    else:
        _emit(openings_markdown(name, summary), args.out)
    return 0


def _cmd_curve(args: argparse.Namespace) -> int:
    from .curve import average_curve, plot_curves

    names = args.names.split(",") if args.names else [p.name for p in args.paths]
    if len(names) != len(args.paths):
        raise UsageError("--names needs one label per path")
    pools = [_load(p, args.field, args.role, args)[1] for p in args.paths]
    max_tokens = args.max_tokens or min(total_tokens(d) for d in pools)
    curves = [
        average_curve(d, max_tokens, name=n, orders=args.orders, seed=args.seed)
        for d, n in zip(pools, names, strict=True)
    ]
    if args.plot is not None:
        plot_curves(curves, args.plot, args.title)
    if args.format == "json":
        _emit(json.dumps([c.to_dict() for c in curves], indent=2), args.out)
    else:
        lines = [
            f"Type-token curves to {max_tokens} tokens, mean of {args.orders} orders.",
            "",
            "| dataset | types at end | Heaps K | Heaps beta |",
            "|---|---:|---:|---:|",
        ]
        for c in curves:
            lines.append(f"| {c.name} | {c.types[-1]:.1f} | {c.k:.3f} | {c.beta:.4f} |")
        _emit("\n".join(lines) + "\n", args.out)
    return 0


COMMANDS = {
    "measure": _cmd_measure,
    "compare": _cmd_compare,
    "openings": _cmd_openings,
    "curve": _cmd_curve,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (UsageError, ValueError, OSError, RuntimeError, csv.Error) as exc:
        print(f"divscore: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
