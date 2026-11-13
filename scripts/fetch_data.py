"""Download the public instruction datasets used in the README and write JSONL.

Nothing downloaded here is committed to the repository. Run:

    uv run --with pyarrow python scripts/fetch_data.py --out data

Each source is pinned to a dataset revision. Parquet files are converted to
JSONL with pyarrow (needed only for this script). Every output line is one
record with ``instruction`` and ``response`` fields.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}"

SOURCES = {
    "dolly": {
        "repo": "databricks/databricks-dolly-15k",
        "rev": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "file": "databricks-dolly-15k.jsonl",
        "licence": "CC BY-SA 3.0",
    },
    "alpaca": {
        "repo": "tatsu-lab/alpaca",
        "rev": "dce01c9b08f87459cf36a430d809084718273017",
        "file": "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
        "licence": "CC BY-NC 4.0",
    },
    "magpie": {
        "repo": "HuggingFaceTB/smoltalk",
        "rev": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
        "file": "data/smol-magpie-ultra/test-00000-of-00001.parquet",
        "licence": "Apache 2.0",
    },
    "personas": {
        "repo": "allenai/tulu-3-sft-personas-instruction-following",
        "rev": "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        "file": "data/train-00000-of-00001.parquet",
        "licence": "ODC-BY",
    },
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)


def first_turn(messages: list[dict], role: str) -> str:
    for m in messages:
        if m.get("role") == role:
            return m.get("content") or ""
    return ""


def records(name: str, raw: Path):
    if raw.suffix == ".jsonl":
        with raw.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                yield {"instruction": row["instruction"], "response": row["response"]}
        return
    import pyarrow.parquet as pq

    table = pq.read_table(raw).to_pylist()
    for row in table:
        if name == "alpaca":
            instr = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
            yield {"instruction": instr, "response": row["output"]}
        else:
            msgs = list(row["messages"])
            yield {
                "instruction": first_turn(msgs, "user"),
                "response": first_turn(msgs, "assistant"),
            }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="directory for downloads and JSONL")
    ap.add_argument("names", nargs="*", default=list(SOURCES), help="subset of sources")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        src = SOURCES[name]
        url = HF.format(**src)
        raw = args.out / f"{name}-raw{Path(src['file']).suffix}"
        print(f"{name}: {src['repo']} ({src['licence']})")
        download(url, raw)
        out = args.out / f"{name}.jsonl"
        n = 0
        with out.open("w", encoding="utf-8") as fh:
            for rec in records(name, raw):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"  wrote {n} records to {out}")


if __name__ == "__main__":
    main()
