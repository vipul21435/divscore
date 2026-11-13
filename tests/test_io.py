import json

import pytest

from divscore.io import detect_format, extract_text, load_texts


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_detect_format():
    assert detect_format("a.jsonl") == "jsonl"
    assert detect_format("a.NDJSON") == "jsonl"
    assert detect_format("a.csv") == "csv"
    assert detect_format("a.tsv") == "tsv"
    assert detect_format("a.json") == "json"
    assert detect_format("notes") == "txt"


def test_jsonl_field_and_skips(tmp_path):
    p = tmp_path / "d.jsonl"
    write_jsonl(p, [{"text": "first"}, {"other": "x"}, {"text": "  "}, {"text": "second"}])
    loaded = load_texts(p)
    assert loaded.texts == ["first", "second"]
    assert loaded.skipped == 2


def test_dotted_and_joined_fields():
    rec = {"instruction": "Sort these", "input": "3 1 2", "meta": {"prompt": "nested"}}
    assert extract_text(rec, field="meta.prompt") == "nested"
    assert extract_text(rec, field="instruction+input") == "Sort these\n3 1 2"
    assert extract_text({"instruction": "Only", "input": ""}, field="instruction+input") == "Only"


def test_chat_role_openai_style(tmp_path):
    p = tmp_path / "chat.jsonl"
    write_jsonl(
        p,
        [
            {
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "bye"},
                ]
            }
        ],
    )
    assert load_texts(p, role="user").texts == ["hi\n\nbye"]
    assert load_texts(p, role="assistant").texts == ["hello"]


def test_chat_role_sharegpt_aliases():
    rec = {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
    assert extract_text(rec, role="user") == "q"
    assert extract_text(rec, role="assistant") == "a"
    assert extract_text(rec, role="system") is None


def test_csv_and_tsv(tmp_path):
    c = tmp_path / "d.csv"
    c.write_text('id,text\n1,"a, b"\n2,c\n', encoding="utf-8")
    assert load_texts(c).texts == ["a, b", "c"]
    t = tmp_path / "d.tsv"
    t.write_text("id\tprompt\n1\tx y\n", encoding="utf-8")
    assert load_texts(t, field="prompt").texts == ["x y"]


def test_plain_text_one_per_line_and_limit(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("one\n\ntwo\nthree\n", encoding="utf-8")
    assert load_texts(p).texts == ["one", "two", "three"]
    assert load_texts(p, limit=2).texts == ["one", "two"]


def test_json_array_and_wrapped(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
    assert load_texts(p).texts == ["a", "b"]
    p.write_text(json.dumps({"rows": [{"text": "c"}]}), encoding="utf-8")
    assert load_texts(p).texts == ["c"]


def test_bad_json_line_reports_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "ok"}\n{bad\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_texts(p)


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        load_texts(tmp_path / "x", fmt="xml")


def test_csv_field_longer_than_default_limit(tmp_path):
    p = tmp_path / "long.csv"
    long_text = "word " * 30_000  # 150,000 characters
    p.write_text(f'text\n"{long_text}"\n', encoding="utf-8")
    assert len(load_texts(p).texts[0]) == len(long_text)


@pytest.mark.parametrize("payload", ["null", "3", '"just a string"'])
def test_json_top_level_must_be_list_or_object(tmp_path, payload):
    p = tmp_path / "d.json"
    p.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="top level"):
        load_texts(p)
