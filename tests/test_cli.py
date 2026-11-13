import json
from pathlib import Path

import pytest

from divscore.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "data"
VARIED = str(EXAMPLES / "varied.jsonl")
TEMPLATED = str(EXAMPLES / "templated.jsonl")
SMALL = ["--budget", "200", "--texts", "20", "--resamples", "10"]


def run_json(tmp_path, args):
    out = tmp_path / "r.json"
    assert main([*args, "--format", "json", "-o", str(out)]) == 0
    return json.loads(out.read_text())


def test_measure_markdown_on_example(capsys):
    assert main(["measure", VARIED, *SMALL]) == 0
    out = capsys.readouterr().out
    assert "## varied.jsonl" in out
    assert "Budget: 200 tokens" in out
    assert "Budget: 20 texts" in out
    assert "| distinct-2 | higher |" in out
    assert "| vendi | higher |" in out
    assert "| self-bleu | lower |" in out


def test_measure_json_shape_and_budget(tmp_path):
    data = run_json(tmp_path, ["measure", TEMPLATED, *SMALL])
    assert data["token_budget"] == 200
    assert data["text_budget"] == 20
    assert data["n_texts"] == 30
    d2 = data["metrics"]["distinct-2"]
    assert d2["budget"] == "tokens"
    assert d2["at_budget"]["lo"] <= d2["at_budget"]["mean"] <= d2["at_budget"]["hi"]
    assert data["metrics"]["vendi"]["budget"] == "texts"
    assert data["metrics"]["vendi"]["naive"] is None
    assert data["metrics"]["opening-top10"]["naive"] is not None


def test_measure_without_text_metrics(tmp_path):
    data = run_json(tmp_path, ["measure", VARIED, "--budget", "200", "--texts", "0"])
    assert "vendi" not in data["metrics"]
    assert data["text_budget"] is None


def test_templated_example_is_less_diverse_at_equal_budget(tmp_path):
    data = run_json(
        tmp_path,
        ["compare", VARIED, TEMPLATED, "--budget", "250", "--texts", "25", "--resamples", "30"],
    )
    diffs = {d["metric"]: d for d in data["differences"]}
    for metric in ("distinct-1", "distinct-2", "entropy-2", "vendi"):
        assert diffs[metric]["diff"] > 0
        assert diffs[metric]["interval_excludes_zero"]
    for metric in ("cr-gzip", "self-bleu", "rouge-l", "opening-top10"):
        assert diffs[metric]["diff"] < 0


def test_compare_markdown_has_verdicts(capsys):
    assert main(["compare", VARIED, TEMPLATED, *SMALL, "--name-a", "v", "--name-b", "t"]) == 0
    out = capsys.readouterr().out
    assert "## v (A) vs t (B)" in out
    assert "| distinct-2 | higher |" in out
    assert "| A |" in out


def test_compare_field_b(tmp_path):
    other = tmp_path / "b.jsonl"
    rows = [json.loads(line) for line in Path(TEMPLATED).read_text().splitlines()]
    other.write_text("\n".join(json.dumps({"prompt": r["text"]}) for r in rows), encoding="utf-8")
    data = run_json(tmp_path, ["compare", VARIED, str(other), "--field-b", "prompt", *SMALL])
    assert data["b"]["n_texts"] == 30


def test_openings_markdown_and_json(tmp_path, capsys):
    assert main(["openings", TEMPLATED, "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "| 1 | write a short | 30 | 1.0000 |" in out
    assert "templates" in out
    data = run_json(tmp_path, ["openings", VARIED, "--top", "5", "--templates", "0"])
    assert data["n_texts"] == 30
    assert len(data["openings"]) == 5
    assert data["templates"] == []


def test_curve_table_json_and_plot(tmp_path, capsys):
    plot = tmp_path / "c.png"
    args = ["curve", VARIED, TEMPLATED, "--names", "v,t", "--orders", "3", "--plot", str(plot)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "| v |" in out and "| t |" in out
    assert plot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = run_json(tmp_path, ["curve", VARIED, "--max-tokens", "200"])
    assert data[0]["tokens"][-1] == 200
    assert 0 < data[0]["heaps_beta"] <= 1.0


def test_curve_names_must_match(capsys):
    assert main(["curve", VARIED, "--names", "a,b"]) == 2
    assert "--names" in capsys.readouterr().err


def test_budget_larger_than_data_is_an_error(capsys):
    assert main(["measure", VARIED, "--budget", "100000"]) == 2
    assert "--budget" in capsys.readouterr().err
    assert main(["measure", VARIED, "--budget", "100", "--texts", "100"]) == 2
    assert "--texts" in capsys.readouterr().err


def test_truncate(tmp_path):
    data = run_json(tmp_path, ["measure", VARIED, *SMALL, "--truncate", "8"])
    assert data["truncate"] == 8
    assert data["lengths"]["max"] == 8


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


def test_csv_error_is_reported_not_raised(tmp_path, capsys, monkeypatch):
    import csv

    import divscore.cli as cli

    def broken(*args, **kwargs):
        raise csv.Error("bad quoting")

    monkeypatch.setattr(cli, "load_texts", broken)
    p = tmp_path / "d.csv"
    p.write_text("text\na\n", encoding="utf-8")
    assert main(["measure", str(p)]) == 2
    assert "bad quoting" in capsys.readouterr().err


def test_json_null_is_an_error(tmp_path, capsys):
    p = tmp_path / "d.json"
    p.write_text("null", encoding="utf-8")
    assert main(["measure", str(p)]) == 2
    assert "top level" in capsys.readouterr().err


def test_keep_case_reaches_the_template_table(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = ["Write A Short Poem About cats"] * 3 + ["write a short poem about dogs"] * 2
    p.write_text("\n".join(json.dumps({"text": r}) for r in rows), encoding="utf-8")
    lower = run_json(tmp_path, ["openings", str(p), "--top", "2"])
    kept = run_json(tmp_path, ["openings", str(p), "--top", "2", "--keep-case"])
    assert lower["templates"][0] == {"text": "write a short poem about", "count": 5, "share": 1.0}
    kept_rows = {row["text"]: row["count"] for row in kept["templates"]}
    assert kept_rows["Write A Short Poem About"] == 3
    assert kept_rows["write a short poem about"] == 2


def test_warns_when_budget_exceeds_half_the_data(capsys):
    assert main(["measure", VARIED, "--budget", "300", "--texts", "0", "--resamples", "2"]) == 0
    assert "warning" in capsys.readouterr().err
    assert main(["measure", VARIED, "--budget", "100", "--texts", "10", "--resamples", "2"]) == 0
    assert "warning" not in capsys.readouterr().err
