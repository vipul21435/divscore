from divscore.text import make_doc, make_docs, tokenize, total_tokens


def test_tokenize_words_and_punctuation():
    toks = tokenize("Hello, world! It's 2 PM.")
    assert toks == ["hello", ",", "world", "!", "it's", "2", "pm", "."]


def test_tokenize_keeps_case_when_asked():
    assert tokenize("Hello World", lowercase=False) == ["Hello", "World"]


def test_tokenize_unicode_words():
    assert tokenize("Grüße aus Köln") == ["grüße", "aus", "köln"]


def test_doc_ends_align_with_text():
    d = make_doc("Write a poem, please.")
    assert d.tokens == ("write", "a", "poem", ",", "please", ".")
    for tok, end in zip(d.tokens, d.ends, strict=True):
        assert d.text[:end].lower().endswith(tok)


def test_truncate_cuts_text_at_token_end():
    d = make_doc("Write a poem about rain.")
    t = d.truncate(3)
    assert t.tokens == ("write", "a", "poem")
    assert t.text == "Write a poem"
    assert len(t) == 3


def test_truncate_zero_and_beyond():
    d = make_doc("one two")
    assert d.truncate(0).text == ""
    assert d.truncate(0).tokens == ()
    assert d.truncate(5) is d


def test_empty_text_gives_empty_doc():
    docs = make_docs(["", "a b"])
    assert len(docs[0]) == 0
    assert total_tokens(docs) == 2
