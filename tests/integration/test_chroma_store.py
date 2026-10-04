"""Chroma vector store integration tests."""

from __future__ import annotations

from pathlib import Path

from tests.retrieval_helpers import make_chunk_record

from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider


def _index_chunks(tmp_path: Path) -> tuple[ChromaVectorStore, FakeEmbeddingProvider, list]:
    index_dir = tmp_path / "chroma_index"
    provider = FakeEmbeddingProvider(vector_dimension=8)
    store = ChromaVectorStore(index_dir=index_dir, collection_name="integration_test")
    chunks = [
        make_chunk_record(chunk_id="a::1", content="delivery delay policy"),
        make_chunk_record(chunk_id="b::1", content="refund timeline policy"),
        make_chunk_record(chunk_id="c::1", content="faq about order status"),
    ]
    vectors = provider.embed_documents([chunk.content for chunk in chunks])
    store.recreate_collection(embedding_dimension=len(vectors[0]))
    store.add_chunks(chunks, vectors)
    return store, provider, chunks


def test_persistent_collection_created(tmp_path: Path) -> None:
    store, _, _ = _index_chunks(tmp_path)
    assert store.count() == 3
    store.close()


def test_count_persists_after_reopen(tmp_path: Path) -> None:
    index_dir = tmp_path / "chroma_index"
    _index_chunks(tmp_path)
    reopened = ChromaVectorStore(index_dir=index_dir, collection_name="integration_test")
    assert reopened.count() == 3
    reopened.close()


def test_metadata_returned_on_search(tmp_path: Path) -> None:
    store, provider, chunks = _index_chunks(tmp_path)
    query_vector = provider.embed_query("delivery delay")
    hits = store.similarity_search(query_vector, k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id in {chunk.chunk_id for chunk in chunks}
    assert hits[0].source_path.startswith("data/02_clean_markdown/")
    store.close()


def test_cosine_ordering_expected(tmp_path: Path) -> None:
    store, provider, _ = _index_chunks(tmp_path)
    query_vector = provider.embed_query("refund timeline policy")
    hits = store.similarity_search(query_vector, k=3)
    similarities = [hit.similarity for hit in hits]
    assert similarities == sorted(similarities, reverse=True)
    assert hits[0].chunk_id == "b::1"
    store.close()


def test_rebuild_clears_old_records(tmp_path: Path) -> None:
    store, provider, chunks = _index_chunks(tmp_path)
    assert store.count() == 3
    store.recreate_collection(embedding_dimension=8)
    single = [chunks[0]]
    vectors = provider.embed_documents([single[0].content])
    store.add_chunks(single, vectors)
    assert store.count() == 1
    store.close()


def _snapshot_index_dir(index_dir: Path) -> dict[str, int]:
    """Return relative file paths and sizes for a stable directory snapshot."""
    if not index_dir.is_dir():
        return {}
    snapshot: dict[str, int] = {}
    for path in sorted(index_dir.rglob("*")):
        if path.is_file():
            snapshot[path.relative_to(index_dir).as_posix()] = path.stat().st_size
    return snapshot


def test_no_project_tree_pollution(tmp_path: Path, project_root: Path) -> None:
    project_index_dir = project_root / "data" / "04_index"
    before = _snapshot_index_dir(project_index_dir)

    index_dir = tmp_path / "chroma_index"
    store, _, _ = _index_chunks(tmp_path)
    store.close()

    temp_files = [path for path in index_dir.rglob("*") if path.is_file()]
    assert temp_files, "expected Chroma index files under temporary directory"
    assert all(path.is_relative_to(tmp_path) for path in temp_files)

    after = _snapshot_index_dir(project_index_dir)
    assert after == before, "Chroma integration test modified project index directory"


def test_engine_failure_on_open_is_a_typed_store_error_not_a_raw_engine_exception(
    tmp_path: Path, monkeypatch
) -> None:
    """A read-only or foreign-owned index fails inside Chroma at open time (it needs write access to
    its SQLite database even to query). The release gate and the UI handle VectorStoreError; a bare
    engine exception would reach the operator as a traceback."""
    import chromadb
    import pytest

    from customer_claims_rag.exceptions import RetrievalError, VectorStoreError

    index_dir = tmp_path / "production_index"
    index_dir.mkdir()
    (index_dir / "chroma.sqlite3").write_bytes(b"")

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("(code: 8) attempt to write a readonly database")

    monkeypatch.setattr(chromadb, "PersistentClient", refuse)
    with pytest.raises(VectorStoreError, match="cannot open the Chroma index") as caught:
        ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims", open_existing=True)
    assert isinstance(caught.value, RetrievalError)
    assert "attempt to write a readonly database" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
