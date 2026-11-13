"""Read texts from JSONL, JSON, CSV/TSV or plain text files.

Records are dictionaries. ``field`` names the text field; a dotted path
(``meta.prompt``) reaches into nested objects, and several fields joined with
``+`` (``instruction+input``) are concatenated with a newline. For chat
records, ``role`` picks the turns written by one speaker from a ``messages``
or ``conversations`` list (OpenAI-style ``role``/``content`` or
ShareGPT-style ``from``/``value``); all turns of that role are joined with a
blank line.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMATS = ("jsonl", "json", "csv", "tsv", "txt")
CHAT_KEYS = ("messages", "conversations")
ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
}


@dataclass(frozen=True)
class Loaded:
    """Texts read from a file and the number of records that had no text."""

    texts: list[str]
    skipped: int


def detect_format(path: str | Path) -> str:
    """Guess the format from the file extension (``txt`` when unknown)."""
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in ("jsonl", "ndjson"):
        return "jsonl"
    if suffix in FORMATS:
        return suffix
    return "txt"


def _get_path(record: Any, dotted: str) -> Any:
    cur = record
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _norm_role(role: Any) -> str:
    r = str(role).strip().lower()
    return ROLE_ALIASES.get(r, r)


def _chat_text(record: Any, field: str | None, role: str) -> str | None:
    turns = _get_path(record, field) if field else None
    if turns is None and isinstance(record, dict):
        for key in CHAT_KEYS:
            if isinstance(record.get(key), list):
                turns = record[key]
                break
    if not isinstance(turns, list):
        return None
    want = _norm_role(role)
    parts: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        who = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value"))
        if who is None or not isinstance(content, str):
            continue
        if _norm_role(who) == want and content.strip():
            parts.append(content)
    return "\n\n".join(parts) if parts else None


def extract_text(record: Any, field: str | None = "text", role: str | None = None) -> str | None:
    """Return the text of one record, or ``None`` when it has none."""
    if role is not None:
        return _chat_text(record, field, role)
    if isinstance(record, str):
        return record if record.strip() else None
    if field is None:
        return None
    parts: list[str] = []
    for name in field.split("+"):
        value = _get_path(record, name.strip())
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(str(value))
    return "\n".join(parts) if parts else None


def iter_records(path: str | Path, fmt: str | None = None) -> Iterator[Any]:
    """Yield raw records (dicts, or strings for plain text) from ``path``."""
    fmt = fmt or detect_format(path)
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")
    p = Path(path)
    if fmt == "json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        yield from data
    elif fmt == "jsonl":
        with p.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{p}:{lineno}: invalid JSON ({exc.msg})") from exc
    elif fmt in ("csv", "tsv"):
        with p.open(encoding="utf-8", newline="") as fh:
            yield from csv.DictReader(fh, delimiter="," if fmt == "csv" else "\t")
    else:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    yield line


def load_texts(
    path: str | Path,
    field: str | None = "text",
    role: str | None = None,
    fmt: str | None = None,
    limit: int | None = None,
) -> Loaded:
    """Read every text from ``path``.

    Records without a usable text are counted in :attr:`Loaded.skipped`.
    ``limit`` stops after that many texts.
    """
    texts: list[str] = []
    skipped = 0
    for record in iter_records(path, fmt):
        text = extract_text(record, field=field, role=role)
        if text is None:
            skipped += 1
            continue
        texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return Loaded(texts, skipped)
