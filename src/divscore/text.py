"""Tokenisation into documents that remember where each token ends.

A :class:`Doc` keeps the raw text next to its tokens so that a document cut to
``k`` tokens still has a matching raw prefix. The byte-level metrics
(compression ratio) use that prefix; the token-level metrics use the tokens.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Words (letters, digits, underscore, with inner apostrophes) or single
# punctuation characters. Unicode-aware through the default ``re`` flags.
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


@dataclass(frozen=True)
class Doc:
    """One text with its tokens and the end offset of each token in ``text``."""

    text: str
    tokens: tuple[str, ...]
    ends: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.tokens)

    def truncate(self, k: int) -> Doc:
        """Return the document cut to its first ``k`` tokens.

        The raw text is cut at the end of token ``k`` so that it stays aligned
        with the tokens.
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        if k >= len(self.tokens):
            return self
        cut = self.ends[k - 1] if k > 0 else 0
        return Doc(self.text[:cut], self.tokens[:k], self.ends[:k])


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split ``text`` into word and punctuation tokens."""
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def make_doc(text: str, lowercase: bool = True) -> Doc:
    """Build a :class:`Doc` from raw text."""
    tokens: list[str] = []
    ends: list[int] = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        tokens.append(tok.lower() if lowercase else tok)
        ends.append(m.end())
    return Doc(text, tuple(tokens), tuple(ends))


def make_docs(texts: Iterable[str], lowercase: bool = True) -> list[Doc]:
    """Tokenise every text; empty texts are kept as empty documents."""
    return [make_doc(t, lowercase=lowercase) for t in texts]


def total_tokens(docs: Iterable[Doc]) -> int:
    """Number of tokens across ``docs``."""
    return sum(len(d) for d in docs)
