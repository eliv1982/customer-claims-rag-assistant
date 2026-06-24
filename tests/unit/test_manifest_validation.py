"""Manifest runtime validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.config import INDEX_FORMAT_VERSION, METADATA_SCHEMA_VERSION
from customer_claims_rag.exceptions import IndexManifestError
from customer_claims_rag.retrieval.manifest import (
    build_manifest,
    validate_manifest_against_runtime,
    validate_query_vector_dimension,
)


def _manifest(**overrides):
    base = {
        "collection_name": "customer_claims",
        "embedding_model": "fake-embedding-model",
        "corpus_fingerprint": "abc",
        "chunk_count": 2,
        "document_count": 1,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "vector_dimension": 8,
    }
    base.update(overrides)
    return build_manifest(**base)


def test_exact_match_passes() -> None:
    manifest = _manifest()
    validate_manifest_against_runtime(
        manifest,
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        chunk_count=2,
        expected_index_format_version=INDEX_FORMAT_VERSION,
        expected_metadata_schema_version=METADATA_SCHEMA_VERSION,
        query_vector_dimension=8,
    )


@pytest.mark.parametrize(
    ("manifest_count", "store_count", "match"),
    [
        (0, 0, "empty index"),
        (0, 3, "empty index"),
        (3, 0, "collection is empty"),
        (3, 2, "chunk count mismatch"),
    ],
)
def test_empty_or_mismatched_chunk_counts_rejected(
    manifest_count: int,
    store_count: int,
    match: str,
) -> None:
    manifest = _manifest(chunk_count=manifest_count)
    with pytest.raises(IndexManifestError, match=match):
        validate_manifest_against_runtime(
            manifest,
            collection_name="customer_claims",
            embedding_model="fake-embedding-model",
            chunk_count=store_count,
        )


def test_matching_nonempty_chunk_counts_accepted() -> None:
    manifest = _manifest(chunk_count=3)
    validate_manifest_against_runtime(
        manifest,
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        chunk_count=3,
    )


def test_empty_consistent_index_rejected_before_search(tmp_path: Path) -> None:
    from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
    from customer_claims_rag.retrieval.manifest import write_manifest_atomic
    from customer_claims_rag.retrieval.retriever import BaselineRetriever

    class RecordingVectorStore:
        def __init__(self) -> None:
            self.searched = False
            self._collection_name = "customer_claims"

        @property
        def collection_name(self) -> str:
            return self._collection_name

        def count(self) -> int:
            return 0

        def similarity_search(self, query_embedding, *, k: int):
            self.searched = True
            return []

        def close(self) -> None:
            pass

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    write_manifest_atomic(
        index_dir,
        _manifest(chunk_count=0, document_count=0),
    )
    store = RecordingVectorStore()
    provider = FakeEmbeddingProvider(model_name="fake-embedding-model", vector_dimension=8)
    retriever = BaselineRetriever(
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
        similarity_threshold=0.0,
    )

    with pytest.raises(IndexManifestError, match="empty index"):
        retriever.validate_index()

    with pytest.raises(IndexManifestError, match="empty index"):
        retriever.search("query")

    assert store.searched is False


def test_metadata_schema_mismatch() -> None:
    manifest = _manifest(metadata_schema_version="9.9.9")
    with pytest.raises(IndexManifestError, match="metadata schema version mismatch"):
        validate_manifest_against_runtime(
            manifest,
            collection_name="customer_claims",
            embedding_model="fake-embedding-model",
            chunk_count=2,
            expected_metadata_schema_version=METADATA_SCHEMA_VERSION,
        )


def test_vector_dimension_mismatch() -> None:
    manifest = _manifest(vector_dimension=8)
    with pytest.raises(IndexManifestError, match="vector dimension mismatch"):
        validate_query_vector_dimension(manifest, 1536)


def test_vector_dimension_checked_before_chroma_query(tmp_path: Path) -> None:
    from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
    from customer_claims_rag.retrieval.manifest import write_manifest_atomic
    from customer_claims_rag.retrieval.models import VectorSearchHit
    from customer_claims_rag.retrieval.retriever import BaselineRetriever

    class RecordingVectorStore:
        def __init__(self) -> None:
            self.searched = False
            self._collection_name = "customer_claims"

        @property
        def collection_name(self) -> str:
            return self._collection_name

        def count(self) -> int:
            return 1

        def similarity_search(self, query_embedding, *, k: int):
            self.searched = True
            return []

        def close(self) -> None:
            pass

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    write_manifest_atomic(
        index_dir,
        _manifest(chunk_count=1, vector_dimension=1536, embedding_model="fake-embedding-model"),
    )
    store = RecordingVectorStore()
    provider = FakeEmbeddingProvider(model_name="fake-embedding-model", vector_dimension=8)
    retriever = BaselineRetriever(
        embedding_provider=provider,
        vector_store=store,
        index_dir=index_dir,
        similarity_threshold=0.0,
    )

    with pytest.raises(IndexManifestError, match="vector dimension mismatch"):
        retriever.search("query")

    assert store.searched is False
