"""Baseline retriever unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.config import DEFAULT_SIMILARITY_THRESHOLD
from customer_claims_rag.exceptions import EmbeddingError, IndexManifestError, RetrievalError
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic
from customer_claims_rag.retrieval.models import VectorSearchHit
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import (
    RetrievalSettings,
    validate_fetch_k,
    validate_similarity_threshold,
    validate_top_k,
)


class StubVectorStore:
    def __init__(self, hits: list[VectorSearchHit], *, collection_name: str = "customer_claims") -> None:
        self._hits = hits
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def recreate_collection(self, *, embedding_dimension: int) -> None:
        pass

    def add_chunks(self, chunks, embeddings) -> None:
        pass

    def count(self) -> int:
        return len(self._hits)

    def similarity_search(self, query_embedding, *, k: int) -> list[VectorSearchHit]:
        return self._hits[:k]

    def close(self) -> None:
        pass


def _hit(
    chunk_id: str,
    similarity: float,
    *,
    content: str = "body",
) -> VectorSearchHit:
    distance = 1.0 - similarity
    return VectorSearchHit(
        chunk_id=chunk_id,
        document_id="doc",
        content=content,
        source_path="data/02_clean_markdown/doc.md",
        chunk_type="policy",
        heading="Heading",
        heading_path=["Heading"],
        distance=distance,
        similarity=similarity,
    )


def _write_manifest(index_dir: Path, *, chunk_count: int, model_name: str) -> None:
    manifest = build_manifest(
        collection_name="customer_claims",
        embedding_model=model_name,
        corpus_fingerprint="abc",
        chunk_count=chunk_count,
        document_count=1,
        metadata_schema_version="1.0.0",
        vector_dimension=8,
    )
    write_manifest_atomic(index_dir, manifest)


def test_default_threshold_is_zero(tmp_path: Path, monkeypatch) -> None:
    from customer_claims_rag import env_bootstrap

    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.delenv("RAG_SIMILARITY_THRESHOLD", raising=False)
    env_bootstrap.reset_project_env()
    env_bootstrap.load_project_env(force=True)
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.0
    settings = RetrievalSettings.from_env()
    assert settings.similarity_threshold == 0.0


def test_default_threshold_returns_candidates_without_explicit_threshold(
    tmp_path: Path,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit("refund::1", 0.6630)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=1, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
    )
    response = retriever.search("query")
    assert len(response.results) == 1
    assert response.similarity_threshold == 0.0


def test_top_k_limit(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit("a::1", 0.95), _hit("b::1", 0.90), _hit("c::1", 0.85)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=3, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        top_k=2,
        fetch_k=3,
        similarity_threshold=0.0,
    )
    response = retriever.search("query")
    assert len(response.results) == 2


def test_fetch_k_used_before_top_k(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit(f"x::{i}", 0.99 - i * 0.01) for i in range(5)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=5, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        top_k=1,
        fetch_k=2,
        similarity_threshold=0.0,
    )
    response = retriever.search("query")
    assert response.candidates_fetched == 2
    assert len(response.results) == 1


def test_threshold_boundary_exactly_070(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit("a::1", 0.70), _hit("b::1", 0.699)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=2, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        top_k=4,
        fetch_k=4,
        similarity_threshold=0.70,
    )
    response = retriever.search("query")
    assert [item.chunk_id for item in response.results] == ["a::1"]


def test_explicit_threshold_070_can_return_empty(
    tmp_path: Path,
    fake_embedding_provider: FakeEmbeddingProvider,
) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit("a::1", 0.6630)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=1, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        similarity_threshold=0.70,
    )
    response = retriever.search("query")
    assert response.results == []
    assert response.candidates_above_threshold == 0


def test_empty_result_with_diagnostics(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit("a::1", 0.50)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=1, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        similarity_threshold=0.70,
    )
    response = retriever.search("query")
    assert response.results == []
    assert response.candidates_fetched == 1
    assert response.candidates_above_threshold == 0


def test_deterministic_tie_break(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit("b::1", 0.90), _hit("a::1", 0.90)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=2, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        similarity_threshold=0.0,
    )
    response = retriever.search("query")
    assert [item.chunk_id for item in response.results] == ["a::1", "b::1"]


def test_empty_query_rejected(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    store = StubVectorStore([])
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
    )
    with pytest.raises(EmbeddingError, match="empty"):
        retriever.search("  ")


def test_invalid_top_k() -> None:
    with pytest.raises(RetrievalError, match="top_k"):
        validate_top_k(0)


def test_invalid_fetch_k() -> None:
    with pytest.raises(RetrievalError, match="fetch_k"):
        validate_fetch_k(2, 4)


def test_invalid_threshold() -> None:
    with pytest.raises(RetrievalError, match="similarity_threshold"):
        validate_similarity_threshold(1.5)


def test_manifest_mismatch_model(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    store = StubVectorStore([_hit("a::1", 0.9)])
    _write_manifest(index_dir, chunk_count=1, model_name="other-model")
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        similarity_threshold=0.0,
    )
    with pytest.raises(IndexManifestError, match="embedding model mismatch"):
        retriever.search("query")


def test_similarity_ordering_descending(tmp_path: Path, fake_embedding_provider: FakeEmbeddingProvider) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hits = [_hit("low", 0.75), _hit("high", 0.95), _hit("mid", 0.85)]
    store = StubVectorStore(hits)
    _write_manifest(index_dir, chunk_count=3, model_name=fake_embedding_provider.model_name)
    retriever = BaselineRetriever(
        embedding_provider=fake_embedding_provider,
        vector_store=store,
        index_dir=index_dir,
        top_k=3,
        fetch_k=3,
        similarity_threshold=0.0,
    )
    response = retriever.search("query")
    similarities = [item.similarity for item in response.results]
    assert similarities == sorted(similarities, reverse=True)
