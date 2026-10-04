"""Embedding provider validation tests."""

from __future__ import annotations

import math
import traceback

import pytest

from customer_claims_rag.cli import search_index as search_cli
from customer_claims_rag.exceptions import EmbeddingError
from customer_claims_rag.input_limits import MAX_CUSTOMER_QUERY_CHARS
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.adapters.openai_embeddings import OpenAIEmbeddingProvider
from customer_claims_rag.retrieval.embedding_validation import (
    MAX_EMBEDDING_BATCH_DOCUMENTS,
    MAX_EMBEDDING_DOCUMENT_CHARS,
    validate_document_texts,
    validate_embedding_batch,
    validate_embedding_vector,
    validate_query_text,
)

SECRET = "sk-test-secret-value"
DOCUMENT_MARKER = "private-document-body"


class _RecordingEmbeddingsClient:
    """OpenAIEmbeddings stand-in that records the production constructor/call contract."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str):
        self.query_calls.append(text)
        return [0.0, 1.0]


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


def test_openai_query_and_document_calls_disable_langchain_tokenizer_preprocessing(
    monkeypatch,
) -> None:
    created: list[_RecordingEmbeddingsClient] = []

    def make_client(**kwargs):
        client = _RecordingEmbeddingsClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.adapters.openai_embeddings.OpenAIEmbeddings",
        make_client,
    )
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", api_key=SECRET)

    assert provider.embed_query("  bounded customer query  ") == [0.0, 1.0]
    assert provider.embed_documents(["bounded canonical chunk"]) == [[1.0, 0.0]]
    assert len(created) == 1
    assert created[0].kwargs == {
        "model": "text-embedding-3-small",
        "api_key": SECRET,
        "check_embedding_ctx_length": False,
    }
    assert created[0].query_calls == ["bounded customer query"]
    assert created[0].document_calls == [["bounded canonical chunk"]]


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


def test_embedding_boundary_enforces_the_accepted_customer_query_limit() -> None:
    assert validate_query_text("x" * MAX_CUSTOMER_QUERY_CHARS) == "x" * MAX_CUSTOMER_QUERY_CHARS
    with pytest.raises(EmbeddingError, match=str(MAX_CUSTOMER_QUERY_CHARS)):
        validate_query_text("x" * (MAX_CUSTOMER_QUERY_CHARS + 1))


@pytest.fixture
def recording_provider(monkeypatch) -> tuple[OpenAIEmbeddingProvider, _RecordingEmbeddingsClient]:
    """Production adapter whose LangChain client is replaced by a call-recording stand-in."""
    created: list[_RecordingEmbeddingsClient] = []

    def make_client(**kwargs):
        client = _RecordingEmbeddingsClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.adapters.openai_embeddings.OpenAIEmbeddings",
        make_client,
    )
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small", api_key=SECRET)
    return provider, created[0]


def test_document_validation_accepts_exactly_the_limit_and_keeps_the_text_verbatim() -> None:
    at_limit = " " + "x" * (MAX_EMBEDDING_DOCUMENT_CHARS - 2) + "\n"
    assert len(at_limit) == MAX_EMBEDDING_DOCUMENT_CHARS
    assert validate_document_texts([at_limit]) == [at_limit]


def test_document_validation_rejects_one_character_over_the_limit_without_echoing_content() -> None:
    oversized = DOCUMENT_MARKER + "x" * (MAX_EMBEDDING_DOCUMENT_CHARS + 1 - len(DOCUMENT_MARKER))
    assert len(oversized) == MAX_EMBEDDING_DOCUMENT_CHARS + 1
    with pytest.raises(EmbeddingError, match=str(MAX_EMBEDDING_DOCUMENT_CHARS)) as exc_info:
        validate_document_texts(["fine", oversized])
    assert "index 1" in str(exc_info.value)
    assert DOCUMENT_MARKER not in str(exc_info.value)


def test_document_validation_batch_boundary() -> None:
    at_limit = ["chunk"] * MAX_EMBEDDING_BATCH_DOCUMENTS
    assert validate_document_texts(at_limit) == at_limit
    with pytest.raises(EmbeddingError, match=str(MAX_EMBEDDING_BATCH_DOCUMENTS)):
        validate_document_texts(at_limit + ["chunk"])


def test_document_validation_of_an_empty_batch_is_empty() -> None:
    assert validate_document_texts([]) == []
    assert validate_document_texts(()) == []


@pytest.mark.parametrize("entry", ["", "   ", "\n\t "])
def test_document_validation_rejects_empty_entries_like_queries(entry: str) -> None:
    with pytest.raises(EmbeddingError, match="index 1.*must not be empty"):
        validate_document_texts(["fine", entry])


@pytest.mark.parametrize("entry", [None, 5, b"bytes", ["nested"]])
def test_document_validation_rejects_non_string_entries(entry: object) -> None:
    with pytest.raises(EmbeddingError, match="index 1.*must be a string"):
        validate_document_texts(["fine", entry])  # type: ignore[list-item]


@pytest.mark.parametrize("batch", ["a single string", b"bytes", None, 7, {"set"}, (c for c in "gen")])
def test_document_validation_rejects_anything_but_a_sequence_of_texts(batch: object) -> None:
    with pytest.raises(EmbeddingError, match="sequence of strings"):
        validate_document_texts(batch)  # type: ignore[arg-type]


def test_adapter_rejects_a_huge_direct_document_before_any_provider_call(recording_provider) -> None:
    provider, client = recording_provider
    with pytest.raises(EmbeddingError, match=str(MAX_EMBEDDING_DOCUMENT_CHARS)):
        provider.embed_documents(["x" * 1_000_000])
    assert client.document_calls == []
    assert client.query_calls == []


def test_adapter_document_boundary_reaches_the_provider_only_at_or_below_the_limit(
    recording_provider,
) -> None:
    provider, client = recording_provider
    at_limit = "x" * MAX_EMBEDDING_DOCUMENT_CHARS
    assert provider.embed_documents([at_limit]) == [[1.0, 0.0]]
    assert client.document_calls == [[at_limit]]
    with pytest.raises(EmbeddingError, match=str(MAX_EMBEDDING_DOCUMENT_CHARS)):
        provider.embed_documents([at_limit + "x"])
    assert client.document_calls == [[at_limit]]


def test_adapter_rejects_the_whole_batch_when_one_document_is_invalid(recording_provider) -> None:
    provider, client = recording_provider
    for bad in ("x" * (MAX_EMBEDDING_DOCUMENT_CHARS + 1), "   ", None, 5):
        with pytest.raises(EmbeddingError):
            provider.embed_documents(["fine", bad, "also fine"])  # type: ignore[list-item]
    assert client.document_calls == []


def test_adapter_rejects_a_bare_string_instead_of_sending_it_character_by_character(
    recording_provider,
) -> None:
    provider, client = recording_provider
    with pytest.raises(EmbeddingError, match="sequence of strings"):
        provider.embed_documents("one document")  # type: ignore[arg-type]
    assert client.document_calls == []


def test_adapter_batch_boundary(recording_provider) -> None:
    provider, client = recording_provider
    at_limit = ["chunk"] * MAX_EMBEDDING_BATCH_DOCUMENTS
    assert len(provider.embed_documents(at_limit)) == MAX_EMBEDDING_BATCH_DOCUMENTS
    assert client.document_calls == [at_limit]
    with pytest.raises(EmbeddingError, match=str(MAX_EMBEDDING_BATCH_DOCUMENTS)):
        provider.embed_documents(at_limit + ["chunk"])
    assert len(client.document_calls) == 1


def test_adapter_empty_batch_makes_no_provider_call(recording_provider) -> None:
    provider, client = recording_provider
    assert provider.embed_documents([]) == []
    assert client.document_calls == []


def test_adapter_sends_the_validated_snapshot_not_the_callers_mutable_list(recording_provider) -> None:
    provider, client = recording_provider
    texts = ["first", "second"]
    provider.embed_documents(texts)
    texts.append("added later")
    assert client.document_calls == [["first", "second"]]


def test_document_and_query_embedding_do_not_need_a_tokenizer(
    recording_provider, monkeypatch, tmp_path
) -> None:
    import tiktoken
    import tiktoken.load

    def forbidden(*args, **kwargs):
        raise AssertionError("embedding input validation must not use a tokenizer or its cache")

    monkeypatch.setattr(tiktoken, "get_encoding", forbidden)
    monkeypatch.setattr(tiktoken, "encoding_for_model", forbidden)
    monkeypatch.setattr(tiktoken.load, "read_file", forbidden)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-tiktoken-cache"))
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(tmp_path / "empty-data-gym-cache"))
    provider, client = recording_provider

    assert validate_document_texts(["bounded canonical chunk"]) == ["bounded canonical chunk"]
    assert provider.embed_documents(["bounded canonical chunk"]) == [[1.0, 0.0]]
    assert provider.embed_query("bounded customer query") == [0.0, 1.0]
    with pytest.raises(EmbeddingError):
        provider.embed_documents(["x" * (MAX_EMBEDDING_DOCUMENT_CHARS + 1)])
    assert client.document_calls == [["bounded canonical chunk"]]
    assert not (tmp_path / "empty-tiktoken-cache").exists()
    assert not (tmp_path / "empty-data-gym-cache").exists()
