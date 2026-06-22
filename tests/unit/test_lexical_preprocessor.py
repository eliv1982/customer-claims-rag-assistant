"""Unit tests for lexical preprocessor."""

from __future__ import annotations

import unicodedata

from customer_claims_rag.retrieval.lexical.preprocessor import normalize_text, tokenize


def test_lowercase() -> None:
    assert tokenize("ВОЗВРАТ") == ["возврат"]


def test_yo_to_e() -> None:
    assert tokenize("ёлка") == ["елка"]
    assert normalize_text("Ёлка") == "елка"


def test_unicode_nfc() -> None:
    composed = "café"
    decomposed = unicodedata.normalize("NFD", composed)
    assert tokenize(composed) == tokenize(decomposed)


def test_cvv_cvc_preserved() -> None:
    assert "cvv" in tokenize("пришлите CVV сюда")
    assert "cvc" in tokenize("код CVC на карте")


def test_punctuation_removed() -> None:
    tokens = tokenize("возврат, деньги! FF-55102")
    assert "возврат" in tokens
    assert "деньги" in tokens
    assert "ff" in tokens
    assert "55102" in tokens


def test_deterministic_output() -> None:
    text = "Для возврата могу прислать CVV — так быстрее?"
    assert tokenize(text) == tokenize(text)


def test_minimum_token_length_two() -> None:
    assert tokenize("a b в") == []


def test_no_stemming() -> None:
    tokens = tokenize("металлическая осколка металл")
    assert "металлическая" in tokens
    assert "осколка" in tokens
    assert "металл" in tokens
