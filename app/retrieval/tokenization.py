"""Tokenization helpers for lexical retrieval."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")


def tokenize(text: str) -> list[str]:
    """Tokenize retrieval text for BM25 and query processing."""

    return TOKEN_PATTERN.findall(text.casefold())
