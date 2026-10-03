"""Embedding provider validation tests."""

from __future__ import annotations

import math
import traceback

import pytest

from customer_claims_rag.cli import search_index as search_cli
from customer_claims_rag.exceptions import EmbeddingError
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.adapters.openai_embeddings import OpenAIEmbeddingProvider
from customer_claims_rag.retrieval.embedding_validation import (
    validate_embedding_batch,
    validate_embedding_vector,
    validate_query_text,
)

SECRET = "sk-test-secret-value"


def test_empty_documents_returns_empty_list(fake_embedding_provider: FakeEmbeddingProvider) -> None:
    assert fake_embedding_provider.embed_documents([]) == []


def test_empty_query_rejected(fake_embedding_provider: FakeEmbeddingProvider) -> None:
    with pytest.raises(EmbeddingError, match="empty"):
        fake_embedding_provider.embed_query("   ")


def test_count_mismatch_rejected() -> None:
    with pytest.raises(EmbeddingError, match="count mismatch"):
        validate_embedding_batch([[0.1, 0.2]], expected_count=2)


def test_dimension_mismatch_rejected() -> None:
    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        validate_embedding_vector([0.1, 0.2], expected_dimension=3)


def test_nan_rejected() -> None:
    with pytest.raises(EmbeddingError, match="NaN"):
        validate_embedding_vector([math.nan, 0.1])


def test_infinity_rejected() -> None:
    with pytest.raises(EmbeddingError, match="NaN"):
        validate_embedding_vector([math.inf, 0.1])


def test_provider_exception_propagation() -> None:
    class BrokenProvider(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            if texts:
                raise RuntimeError("provider down")
            return []

    provider = BrokenProvider()
    with pytest.raises(RuntimeError, match="provider down"):
        provider.embed_documents(["hello"])


def test_openai_missing_api_key_message() -> None:
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(model_name="text-embedding-3-small", api_key=None)


def test_openai_api_key_not_in_error_message(monkeypatch) -> None:
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", api_key=SECRET)

    class FakeClient:
        def embed_query(self, text: str):
            raise RuntimeError(f"Invalid API key {SECRET} provided")

        def embed_documents(self, texts):
            raise RuntimeError(f"Authorization failed for {SECRET}")

    monkeypatch.setattr(provider, "_client", FakeClient())
    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_query("hello")
    exc = exc_info.value
    assert SECRET not in str(exc)
    assert SECRET not in repr(exc)
    assert exc.__cause__ is None
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert SECRET not in formatted


def test_openai_documents_error_sanitized(monkeypatch) -> None:
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", api_key=SECRET)

    class FakeClient:
        def embed_documents(self, texts):
            raise RuntimeError(f"provider leaked {SECRET}")

    monkeypatch.setattr(provider, "_client", FakeClient())
    with pytest.raises(EmbeddingError) as exc_info:
        provider.embed_documents(["hello"])
    exc = exc_info.value
    assert SECRET not in str(exc)
    assert exc.__cause__ is None


def test_search_cli_does_not_leak_secret_without_verbose(temp_project: Path, monkeypatch, capsys) -> None:
    def leaking_factory(*, model_name: str, api_key: str | None = None):
        raise EmbeddingError("OpenAI embedding request failed")

    monkeypatch.setattr(search_cli, "create_embedding_provider", leaking_factory)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.chdir(temp_project)
    code = search_cli.main(["query", "--index-dir", "data/04_index"])
    captured = capsys.readouterr()
    assert code != 0
    assert SECRET not in captured.err
    assert SECRET not in captured.out
    assert "Traceback" not in captured.err


def test_search_cli_does_not_leak_secret_with_verbose(temp_project: Path, monkeypatch, capsys) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError(f"boom {SECRET}")

    monkeypatch.setattr(search_cli, "run_search", explode)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.chdir(temp_project)
    code = search_cli.main(["query", "--verbose"])
    captured = capsys.readouterr()
    assert code != 0
    assert SECRET not in captured.err
    assert "Traceback" in captured.err


def test_same_text_same_vector(fake_embedding_provider: FakeEmbeddingProvider) -> None:
    first = fake_embedding_provider.embed_query("stable text")
    second = fake_embedding_provider.embed_query("stable text")
    assert first == second


def test_validate_query_text_strips() -> None:
    assert validate_query_text("  hello  ") == "hello"
