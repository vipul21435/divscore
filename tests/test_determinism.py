import json
import os
import random
import subprocess
import sys

import pytest


def run_cli(args, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    proc = subprocess.run(
        [sys.executable, "-m", "divscore.cli", *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


@pytest.fixture
def tied_corpus(tmp_path):
    # Every 5-gram occurs in exactly one text, so the ten "most common"
    # templates are a tie between all of them.
    rng = random.Random(0)
    path = tmp_path / "tied.jsonl"

    def word():
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))

    rows = [" ".join(word() for _ in range(7)) for _ in range(30)]
    path.write_text("\n".join(json.dumps({"text": r}) for r in rows), encoding="utf-8")
    return str(path)


def test_measure_is_independent_of_hash_seed(tied_corpus):
    args = ["measure", tied_corpus, "--budget", "50", "--texts", "30", "--resamples", "5"]
    args += ["--format", "json"]
    a = run_cli(args, 0)
    b = run_cli(args, 1)
    assert a == b


def test_openings_templates_are_independent_of_hash_seed(tied_corpus):
    args = ["openings", tied_corpus, "--templates", "5", "--format", "json"]
    a = run_cli(args, 1)
    b = run_cli(args, 2)
    assert a["templates"] == b["templates"]
    assert a == b
