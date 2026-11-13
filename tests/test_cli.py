import json
from pathlib import Path

import pytest

from divscore.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "data"


def test_measure_markdown_on_example(capsys):
    rc = main(["measure", str(EXAMPLES / "varied.jsonl"), "--budget", "200", "--resamples", "20"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## varied.jsonl" in out
    assert "| distinct-2 | higher |" in out
    assert "| cr-gzip | lower |" in out


def test_measure_json_shape_and_budget(tmp_path):
    out = tmp_path / "r.json"
    rc = main(
        [
            "measure",
            str(EXAMPLES / "templated.jsonl"),
            "--budget",
            "250",
            "--resamples",
            "10",
            "--format",
            "json",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["budget"] == 250
    assert data["n_texts"] == 30
    d2 = data["metrics"]["distinct-2"]
    assert d2["higher_is_more_diverse"] is True
    assert d2["at_budget"]["lo"] <= d2["at_budget"]["mean"] <= d2["at_budget"]["hi"]


def test_templated_example_is_less_diverse_at_equal_budget(tmp_path):
    reports = {}
    for name in ("varied", "templated"):
        out = tmp_path / f"{name}.json"
        main(
            [
                "measure",
                str(EXAMPLES / f"{name}.jsonl"),
                "--budget",
                "250",
                "--resamples",
                "30",
                "--format",
                "json",
                "-o",
                str(out),
            ]
        )
        reports[name] = json.loads(out.read_text())["metrics"]
    for metric in ("distinct-1", "distinct-2", "entropy-2"):
        assert (
            reports["varied"][metric]["at_budget"]["mean"]
            > reports["templated"][metric]["at_budget"]["mean"]
        )
    assert (
        reports["varied"]["cr-gzip"]["at_budget"]["mean"]
        < reports["templated"]["cr-gzip"]["at_budget"]["mean"]
    )


def test_budget_larger_than_data_is_an_error(capsys):
    rc = main(["measure", str(EXAMPLES / "varied.jsonl"), "--budget", "100000"])
    assert rc == 2
    assert "--budget" in capsys.readouterr().err


def test_no_texts_is_an_error(tmp_path, capsys):
    p = tmp_path / "e.jsonl"
    p.write_text('{"other": 1}\n', encoding="utf-8")
    assert main(["measure", str(p)]) == 2
    assert "no texts" in capsys.readouterr().err


def test_missing_file_is_an_error(tmp_path, capsys):
    assert main(["measure", str(tmp_path / "nope.jsonl")]) == 2
    assert "divscore:" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "divscore" in capsys.readouterr().out
