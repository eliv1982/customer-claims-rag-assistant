"""Index builder manifest lifecycle regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.retrieval_helpers import make_chunk_record

from customer_claims_rag.exceptions import IndexBuildError, IndexManifestError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval import index_builder as index_builder_module
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.manifest import load_manifest, manifest_path
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.token_counter import TiktokenCounter


class FailingAddVectorStore(ChromaVectorStore):
    def add_chunks(self, chunks, embeddings):
        raise RuntimeError("store add failed")


class WrongCountVectorStore(ChromaVectorStore):
    def count(self) -> int:
        return 0


def _builder(
    tmp_path: Path,
    *,
    store: ChromaVectorStore | None = None,
    corpus_builder: CorpusBuilder | None = None,
) -> IndexBuilder:
    index_dir = tmp_path / "data" / "04_index"
    provider = FakeEmbeddingProvider(model_name="fake-embedding-model", vector_dimension=8)
    vector_store = store or ChromaVectorStore(
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
        vector_store=vector_store,
        index_dir=index_dir,
        batch_size=2,
    )


def _patch_corpus(monkeypatch, corpus_builder: CorpusBuilder, chunks: list) -> None:
    monkeypatch.setattr(
        corpus_builder,
        "build_from_directory",
        lambda _input_dir: ([], chunks),
    )


def test_embedding_failure_preserves_existing_manifest(tmp_path: Path, monkeypatch) -> None:
    chunk_a = make_chunk_record(chunk_id="doc::chunk-001", content="build A content")
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    index_builder = _builder(tmp_path, corpus_builder=corpus_builder)
    _patch_corpus(monkeypatch, corpus_builder, [chunk_a])

    report_a = index_builder.build_from_directory(
        tmp_path / "data" / "02_clean_markdown",
        rebuild=True,
    )
    manifest_a = load_manifest(index_builder.index_dir)

    class ExplodingProvider(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            if texts:
                raise RuntimeError("embedding failed")
            return []

    chunk_b = make_chunk_record(chunk_id="doc::chunk-001", content="build B content")
    _patch_corpus(monkeypatch, corpus_builder, [chunk_b])
    index_builder.embedding_provider = ExplodingProvider(
        model_name="fake-embedding-model",
        vector_dimension=8,
    )

    with pytest.raises(IndexBuildError, match="embedding batch failed"):
        index_builder.build_from_directory(
            tmp_path / "data" / "02_clean_markdown",
            rebuild=True,
        )

    manifest_after = load_manifest(index_builder.index_dir)
    assert manifest_after.corpus_fingerprint == manifest_a.corpus_fingerprint
    assert manifest_after.corpus_fingerprint == report_a.fingerprint


def test_invalidation_failure_preserves_manifest_and_skips_recreate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chunk = make_chunk_record()
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    store = ChromaVectorStore(
        index_dir=tmp_path / "data" / "04_index",
        collection_name="test_collection",
    )
    index_builder = _builder(tmp_path, store=store, corpus_builder=corpus_builder)
    _patch_corpus(monkeypatch, corpus_builder, [chunk])
    index_builder.build_from_directory(tmp_path / "data" / "02_clean_markdown", rebuild=True)
    assert manifest_path(index_builder.index_dir).is_file()

    recreate_called = False

    def spy_recreate(*, embedding_dimension: int) -> None:
        nonlocal recreate_called
        recreate_called = True

    monkeypatch.setattr(store, "recreate_collection", spy_recreate)
    monkeypatch.setattr(
        index_builder_module,
        "invalidate_manifest",
        lambda _index_dir: (_ for _ in ()).throw(
            IndexBuildError("failed to invalidate existing index manifest")
        ),
    )

    with pytest.raises(IndexBuildError, match="failed to invalidate"):
        index_builder.build_from_directory(
            tmp_path / "data" / "02_clean_markdown",
            rebuild=True,
        )

    assert recreate_called is False
    assert manifest_path(index_builder.index_dir).is_file()


def test_failed_manifest_write_removes_stale_manifest(tmp_path: Path, monkeypatch) -> None:
    chunk_a = make_chunk_record(chunk_id="doc::chunk-001", content="build A content")
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    index_builder = _builder(tmp_path, corpus_builder=corpus_builder)
    _patch_corpus(monkeypatch, corpus_builder, [chunk_a])
    report_a = index_builder.build_from_directory(
        tmp_path / "data" / "02_clean_markdown",
        rebuild=True,
    )
    assert report_a.fingerprint

    chunk_b = make_chunk_record(chunk_id="doc::chunk-001", content="build B content")
    _patch_corpus(monkeypatch, corpus_builder, [chunk_b])

    def fail_write(_index_dir, _manifest) -> None:
        raise OSError("manifest write failed")

    monkeypatch.setattr(index_builder_module, "write_manifest_atomic", fail_write)

    with pytest.raises(IndexBuildError):
        index_builder.build_from_directory(
            tmp_path / "data" / "02_clean_markdown",
            rebuild=True,
        )

    assert not manifest_path(index_builder.index_dir).exists()

    retriever = BaselineRetriever(
        embedding_provider=index_builder.embedding_provider,
        vector_store=index_builder.vector_store,
        index_dir=index_builder.index_dir,
        similarity_threshold=0.0,
    )
    with pytest.raises(IndexManifestError, match="manifest not found"):
        retriever.search("query")


def test_add_failure_leaves_no_manifest(tmp_path: Path, monkeypatch) -> None:
    chunk = make_chunk_record()
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    failing_store = FailingAddVectorStore(
        index_dir=tmp_path / "data" / "04_index",
        collection_name="test_collection",
    )
    index_builder = _builder(tmp_path, store=failing_store, corpus_builder=corpus_builder)
    _patch_corpus(monkeypatch, corpus_builder, [chunk])

    with pytest.raises(IndexBuildError, match="index rebuild failed"):
        index_builder.build_from_directory(
            tmp_path / "data" / "02_clean_markdown",
            rebuild=True,
        )

    assert not manifest_path(index_builder.index_dir).exists()


def test_count_mismatch_leaves_no_manifest(tmp_path: Path, monkeypatch) -> None:
    chunk = make_chunk_record()
    corpus_builder = CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=tmp_path.resolve(),
    )
    wrong_count_store = WrongCountVectorStore(
        index_dir=tmp_path / "data" / "04_index",
        collection_name="test_collection",
    )
    index_builder = _builder(tmp_path, store=wrong_count_store, corpus_builder=corpus_builder)
    _patch_corpus(monkeypatch, corpus_builder, [chunk])

    with pytest.raises(IndexBuildError, match="chunk count mismatch"):
        index_builder.build_from_directory(
            tmp_path / "data" / "02_clean_markdown",
            rebuild=True,
        )

    assert not manifest_path(index_builder.index_dir).exists()
