"""Unit tests for manifest-driven lexical corpus loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from customer_claims_rag.config import METADATA_SCHEMA_VERSION
from customer_claims_rag.exceptions import DuplicateChunkIdError, IndexManifestError
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.lexical.corpus_loader import load_lexical_corpus_from_chroma
from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic
from tests.retrieval_helpers import make_chunk_record

EXPANDED_CORPUS_CHUNK_COUNT = 333


def _write_index(
    tmp_path: Path,
    *,
    chunks,
    chunk_count: int | None = None,
    corpus_fingerprint: str = "corpus-loader-test-fingerprint",
    document_count: int | None = None,
) -> Path:
    index_dir = tmp_path / "index"
    provider = FakeEmbeddingProvider(vector_dimension=8)
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    store.recreate_collection(embedding_dimension=provider.vector_dimension)
    vectors = provider.embed_documents([chunk.content for chunk in chunks])
    store.add_chunks(chunks, vectors)
    store.close()
    resolved_chunk_count = chunk_count if chunk_count is not None else len(chunks)
    resolved_document_count = (
        document_count
        if document_count is not None
        else len({chunk.document_id for chunk in chunks})
    )
    write_manifest_atomic(
        index_dir,
        build_manifest(
            collection_name="customer_claims",
            embedding_model=provider.model_name,
            corpus_fingerprint=corpus_fingerprint,
            chunk_count=resolved_chunk_count,
            document_count=resolved_document_count,
            metadata_schema_version=METADATA_SCHEMA_VERSION,
            vector_dimension=provider.vector_dimension,
        ),
    )
    return index_dir


def test_loads_manifest_driven_corpus_with_333_rows(tmp_path: Path) -> None:
    chunks = [
        make_chunk_record(
            chunk_id=f"doc-{index:03d}::chunk-001",
            document_id=f"doc-{index:03d}",
        )
        for index in range(EXPANDED_CORPUS_CHUNK_COUNT)
    ]
    index_dir = _write_index(tmp_path, chunks=chunks)
    loaded = load_lexical_corpus_from_chroma(
        index_dir=index_dir,
        collection_name="customer_claims",
    )
    assert len(loaded) == EXPANDED_CORPUS_CHUNK_COUNT
    assert loaded[0].chunk_id == "doc-000::chunk-001"


def test_rejects_manifest_collection_count_mismatch(tmp_path: Path) -> None:
    chunks = [make_chunk_record(chunk_id="doc::chunk-001")]
    index_dir = _write_index(tmp_path, chunks=chunks, chunk_count=EXPANDED_CORPUS_CHUNK_COUNT)
    with pytest.raises(IndexManifestError, match="chunk count mismatch"):
        load_lexical_corpus_from_chroma(
            index_dir=index_dir,
            collection_name="customer_claims",
        )


def test_rejects_duplicate_chunk_ids(tmp_path: Path) -> None:
    chunk = make_chunk_record(chunk_id="dup::chunk-001")
    index_dir = _write_index(tmp_path, chunks=[chunk], chunk_count=2)
    duplicate_metadata = {
        "chunk_id": "dup::chunk-001",
        "document_id": chunk.document_id,
        "heading": chunk.heading,
        "chunk_type": chunk.strategy,
        "source_path": chunk.source_path,
    }
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.get.return_value = {
        "ids": ["dup::chunk-001", "dup::chunk-001"],
        "documents": [chunk.content, chunk.content],
        "metadatas": [duplicate_metadata, duplicate_metadata],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    with patch("chromadb.PersistentClient", return_value=mock_client):
        with pytest.raises(DuplicateChunkIdError, match="duplicate chunk_id"):
            load_lexical_corpus_from_chroma(
                index_dir=index_dir,
                collection_name="customer_claims",
            )


def test_enforces_expected_fingerprint_when_supplied(tmp_path: Path) -> None:
    chunks = [make_chunk_record(chunk_id="doc::chunk-001")]
    index_dir = _write_index(
        tmp_path,
        chunks=chunks,
        corpus_fingerprint="actual-fingerprint",
    )
    with pytest.raises(IndexManifestError, match="corpus fingerprint mismatch"):
        load_lexical_corpus_from_chroma(
            index_dir=index_dir,
            collection_name="customer_claims",
            expected_fingerprint="expected-fingerprint",
        )


def test_accepts_matching_expected_fingerprint(tmp_path: Path) -> None:
    chunks = [make_chunk_record(chunk_id="doc::chunk-001")]
    fingerprint = "matching-fingerprint"
    index_dir = _write_index(tmp_path, chunks=chunks, corpus_fingerprint=fingerprint)
    loaded = load_lexical_corpus_from_chroma(
        index_dir=index_dir,
        collection_name="customer_claims",
        expected_fingerprint=fingerprint,
    )
    assert len(loaded) == 1


def test_rejects_empty_manifest_chunk_count(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path, chunks=[], chunk_count=0, document_count=0)
    with pytest.raises(IndexManifestError, match="empty index"):
        load_lexical_corpus_from_chroma(
            index_dir=index_dir,
            collection_name="customer_claims",
        )
