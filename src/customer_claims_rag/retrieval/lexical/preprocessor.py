"""Deterministic lexical tokenizer for hybrid retrieval experiments."""

from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = "tokenizer-v1"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}|[а-я]{2,}|cvv|cvc", re.IGNORECASE)


def normalize_text(value: str) -> str:
    """Apply Unicode NFC, lowercase, and ё→е normalization."""
    normalized = unicodedata.normalize("NFC", value)
    return normalized.replace("ё", "е").replace("Ё", "Е").lower()


def tokenize(value: str) -> list[str]:
    """Tokenize text into deterministic lexical tokens."""
    normalized = normalize_text(value)
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(normalized)]
