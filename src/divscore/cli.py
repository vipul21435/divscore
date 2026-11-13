"""Command-line interface: ``divscore measure``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .io import FORMATS, load_texts
from .measure import measure
from .report import measurement_json, measurement_markdown
from .text import make_docs, total_tokens

DEFAULT_BUDGET = 10_000


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--field", default="text", help="text field; dotted path, join with + (default: text)"
    )
    p.add_argument("--role", default=None, help="for chat records: take the turns of this role")
    p.add_argument("--input-format", choices=FORMATS, default=None, help="default: from extension")
    p.add_argument("--limit", type=int, default=None, help="read at most this many texts")
    p.add_argument("--keep-case", action="store_true", help="do not lowercase tokens")


def _add_budget_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"tokens per subsample (default: {DEFAULT_BUDGET})",
    )
    p.add_argument("--resamples", type=int, default=100, help="number of subsamples (default: 100)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    p.add_argument("--level", type=float, default=0.95, help="interval level (default: 0.95)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divscore",
        description="Diversity of a text dataset, with every metric taken at a fixed token budget.",
    )
    parser.add_argument("--version", action="version", version=f"divscore {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="measure one dataset")
    m.add_argument("path", type=Path)
    m.add_argument("--name", default=None, help="label in the report (default: file name)")
    _add_input_args(m)
    _add_budget_args(m)
    m.add_argument("--format", choices=("md", "json"), default="md", help="report format")
    m.add_argument("-o", "--out", type=Path, default=None, help="write the report here")
    return parser


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    else:
        out.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _cmd_measure(args: argparse.Namespace) -> int:
    loaded = load_texts(
        args.path, field=args.field, role=args.role, fmt=args.input_format, limit=args.limit
    )
    if not loaded.texts:
        print(f"divscore: no texts found in {args.path} (field {args.field!r})", file=sys.stderr)
        return 2
    docs = make_docs(loaded.texts, lowercase=not args.keep_case)
    available = total_tokens(docs)
    if available < args.budget:
        print(
            f"divscore: {args.path} holds {available} tokens, fewer than the budget of "
            f"{args.budget}; pass --budget {available} or less",
            file=sys.stderr,
        )
        return 2
    result = measure(
        docs,
        budget=args.budget,
        name=args.name or args.path.name,
        resamples=args.resamples,
        seed=args.seed,
        level=args.level,
        skipped=loaded.skipped,
    )
    text = measurement_json(result) if args.format == "json" else measurement_markdown(result)
    _emit(text, args.out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "measure":
            return _cmd_measure(args)
    except (ValueError, OSError) as exc:
        print(f"divscore: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
