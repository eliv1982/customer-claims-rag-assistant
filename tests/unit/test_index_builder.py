"""Index builder unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.retrieval_helpers import make_chunk_record

from customer_claims_rag.exceptions import IndexBuildError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.fingerprint import sort_chunks_deterministic
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.manifest import load_manifest, manifest_path
from customer_claims_rag.token_counter import TiktokenCounter

BASELINE_CHUNK_COUNT = 215
RELEASE_EXPANSION_DOC11_CHUNK_COUNT = 20
RELEASE_EXPANSION_DOC12_CHUNK_COUNT = 23
RELEASE_EXPANSION_DOC13_CHUNK_COUNT = 22
EXPECTED_CORPUS_CHUNK_COUNT = (
    BASELINE_CHUNK_COUNT
    + RELEASE_EXPANSION_DOC11_CHUNK_COUNT
    + RELEASE_EXPANSION_DOC12_CHUNK_COUNT
    + RELEASE_EXPANSION_DOC13_CHUNK_COUNT
)


class ExplodingEmbeddingProvider(FakeEmbeddingProvider):
    def embed_documents(self, texts):
        if texts:
            raise RuntimeError("embedding failed")
        return []


class FailingAddVectorStore(ChromaVectorStore):
    def add_chunks(self, chunks, embeddings):
        raise RuntimeError("store add failed")


def _builder(
    tmp_path: Path,
    *,
    embedding_provider: FakeEmbeddingProvider | None = None,
    vector_store: ChromaVectorStore | None = None,
    corpus_builder: CorpusBuilder | None = None,
) -> IndexBuilder:
    index_dir = tmp_path / "data" / "04_index"
    provider = embedding_provider or FakeEmbeddingProvider()
    store = vector_store or ChromaVectorStore(
        index_dir=index_dir,
        collection_name="test_collection",
    )
    builder = corpus_builder or CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    return IndexBuilder(
        corpus_builder=builder,
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
        batch_size=2,
    )


def test_zero_chunks_error(tmp_path: Path, monkeypatch) -> None:
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    monkeypatch.setattr(
        corpus_builder,
        "build_from_directory",
        lambda _input_dir: ([], []),
    )
    index_builder = _builder(tmp_path, corpus_builder=corpus_builder)
    with pytest.raises(IndexBuildError, match="no chunks"):
        index_builder.build_from_directory(tmp_path / "data" / "02_clean_markdown", rebuild=True)


def test_duplicate_ids_error(tmp_path: Path, monkeypatch) -> None:
    duplicate = make_chunk_record(chunk_id="dup::chunk-001")
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    monkeypatch.setattr(
        corpus_builder,
        "build_from_directory",
        lambda _input_dir: ([], [duplicate, duplicate]),
    )
    index_builder = _builder(tmp_path, corpus_builder=corpus_builder)
    with pytest.raises(IndexBuildError, match="duplicate chunk_id"):
        index_builder.build_from_directory(tmp_path / "data" / "02_clean_markdown", rebuild=True)


def test_deterministic_ordering() -> None:
    chunks = [
        make_chunk_record(chunk_id="b::chunk-001"),
        make_chunk_record(chunk_id="a::chunk-001"),
    ]
    ordered = sort_chunks_deterministic(chunks)
    assert [chunk.chunk_id for chunk in ordered] == ["a::chunk-001", "b::chunk-001"]


def test_rebuild_count_matches(temp_project: Path, builder: CorpusBuilder) -> None:
    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    index_builder = IndexBuilder(
        corpus_builder=builder,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        index_dir=index_dir,
        batch_size=32,
    )
    input_dir = temp_project / "data" / "02_clean_markdown"
    first = index_builder.build_from_directory(input_dir, rebuild=True)
    assert first.chunks == EXPECTED_CORPUS_CHUNK_COUNT
    assert store.count() == EXPECTED_CORPUS_CHUNK_COUNT
    second = index_builder.build_from_directory(input_dir, rebuild=True)
    assert second.chunks == EXPECTED_CORPUS_CHUNK_COUNT
    assert store.count() == EXPECTED_CORPUS_CHUNK_COUNT
    assert first.fingerprint == second.fingerprint


def test_manifest_written_only_after_success(temp_project: Path, builder: CorpusBuilder) -> None:
    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    index_builder = IndexBuilder(
        corpus_builder=builder,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        index_dir=index_dir,
    )
    report = index_builder.build_from_directory(
        temp_project / "data" / "02_clean_markdown",
        rebuild=True,
    )
    manifest = load_manifest(index_dir)
    assert manifest.chunk_count == report.chunks
    assert manifest.corpus_fingerprint == report.fingerprint


def test_failed_embedding_leaves_no_manifest(tmp_path: Path, monkeypatch) -> None:
    chunk = make_chunk_record()
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    monkeypatch.setattr(
        corpus_builder,
        "build_from_directory",
        lambda _input_dir: ([], [chunk]),
    )
    index_builder = _builder(
        tmp_path,
        corpus_builder=corpus_builder,
        embedding_provider=ExplodingEmbeddingProvider(),
    )
    with pytest.raises(IndexBuildError, match="embedding batch failed"):
        index_builder.build_from_directory(tmp_path / "data" / "02_clean_markdown", rebuild=True)
    assert not manifest_path(index_builder.index_dir).exists()


def test_failed_store_add_leaves_no_manifest(tmp_path: Path, monkeypatch) -> None:
    chunk = make_chunk_record()
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    monkeypatch.setattr(
        corpus_builder,
        "build_from_directory",
        lambda _input_dir: ([], [chunk]),
    )
    failing_store = FailingAddVectorStore(
        index_dir=tmp_path / "data" / "04_index",
        collection_name="test_collection",
    )
    index_builder = _builder(
        tmp_path,
        corpus_builder=corpus_builder,
        vector_store=failing_store,
    )
    with pytest.raises(IndexBuildError, match="index rebuild failed"):
        index_builder.build_from_directory(tmp_path / "data" / "02_clean_markdown", rebuild=True)
    assert not manifest_path(index_builder.index_dir).exists()


def test_corrupted_manifest_detection(tmp_path: Path) -> None:
    from customer_claims_rag.exceptions import IndexManifestError
    from customer_claims_rag.retrieval.manifest import load_manifest

    index_dir = tmp_path / "data" / "04_index"
    index_dir.mkdir(parents=True)
    manifest_path(index_dir).write_text("{not-json", encoding="utf-8")
    with pytest.raises(IndexManifestError, match="corrupted JSON"):
        load_manifest(index_dir)
